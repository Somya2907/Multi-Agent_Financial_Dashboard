"""Multi-HyDE pre-retrieval module.

Pipeline
--------
    original query
        │
        ▼  classify_query()  →  GENERAL | NUMERICAL
        │
        ▼  (LLM)  generate_diverse_queries  [type-aware prompt]
    [q1, q2, q3, q4]
        │
        ▼  (LLM, parallel)  generate_hypothetical_doc × N  [type-aware prompt]
    [doc1, doc2, doc3, doc4]
        │
        ▼  Titan embed_batch
    [vec1 … vecN]
        │
        ├── N × FAISS dense search    (dense_weight per list)
        ├── N × BM25 sparse search    (sparse_weight per list)
        │   + decomposed sub-queries  (NUMERICAL only, BM25 only)
        │
        ▼  weighted multi-list RRF
    merged pool
        │
        ▼  BGE rerank with ORIGINAL query
    final ranked chunks

Key design choices
------------------
- Original query is NEVER embedded — only hypothetical documents are.
- Original query IS used for BM25 (keyword match) and BGE reranker.
- NUMERICAL queries: sparse_weight=0.6 / dense_weight=0.4 (keyword terms dominate).
- NUMERICAL queries: metric-specific prompts generate balance-sheet-style passages
  with numbers ("current assets of $X, current liabilities of $Y, ratio = Z").
- NUMERICAL queries: automatic sub-query decomposition adds component keyword
  queries to the BM25 pool ("current assets", "current liabilities", etc.).
"""

import json
import logging
import concurrent.futures
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

# Must import reranker (and torch) at module level so PyTorch's OMP runtime
# initializes before faiss loads its own copy — prevents macOS segfault.
from pipeline.reasoning.reranker import rerank as _rerank

if TYPE_CHECKING:
    from openai import OpenAI
    from pipeline.indexing.faiss_manager import FAISSManager
    from pipeline.indexing.bm25_index import BM25Index
    from pipeline.embedding.embedder import TitanEmbedder

logger = logging.getLogger(__name__)


# ── Query type classification ─────────────────────────────────────────────────

class QueryType(Enum):
    GENERAL   = "general"
    NUMERICAL = "numerical"


# Keywords that signal a numerical / ratio / balance-sheet query
_NUMERICAL_SIGNALS = {
    # ratio names
    "current ratio", "debt-to-equity", "debt to equity", "quick ratio",
    "price to earnings", "p/e ratio", "return on equity", "roe", "roa",
    "gross margin", "operating margin", "net margin", "ebitda margin",
    # balance-sheet terms
    "current assets", "current liabilities", "total assets", "total liabilities",
    "shareholder equity", "stockholder equity", "working capital",
    # action words that imply calculation
    "calculate", "compute", "rank", "ranking", "highest", "lowest",
    "compare", "comparison", "which company", "which companies",
    # financial statement items
    "revenue", "net income", "earnings", "cash flow", "free cash flow",
    "liabilities", "assets", "equity", "balance sheet",
    # numerical signal words
    "ratio", "margin", "multiple", "percentage", "%",
    "liquidity", "leverage", "solvency",
}


def classify_query(query: str) -> QueryType:
    """Return NUMERICAL if the query targets metrics, ratios, or balance-sheet data.

    Uses fast keyword matching — no LLM call.
    """
    q = query.lower()
    for signal in _NUMERICAL_SIGNALS:
        if signal in q:
            logger.info(f"[HyDE] Query classified as NUMERICAL (matched: '{signal}')")
            return QueryType.NUMERICAL
    logger.info("[HyDE] Query classified as GENERAL")
    return QueryType.GENERAL


# ── Intent classification for section-aware reranking ────────────────────────
#
# A finer-grained signal than QueryType: tells the reranker *which* document
# sections / source types to prefer when scores are close.  Evaluated in
# priority order (sentiment > risk > metric > general) so that a query like
# "what did management say about liquidity risk" is treated as sentiment
# (transcript-heavy) rather than metric.

_SENTIMENT_SIGNALS = {
    "management", "mgmt", "ceo", "cfo", "said", "stated", "described",
    "commentary", "guidance", "outlook", "tone", "earnings call",
    "prepared remarks", "q&a", "transcript", "forward-looking",
    "sentiment", "ramp", "announced",
}

_RISK_SIGNALS = {
    "risk", "risks", "litigation", "lawsuit", "legal proceedings",
    "regulatory", "regulation", "antitrust", "cybersecurity", "privacy",
    "compliance", "supply chain", "geopolitical", "interest rate",
    "foreign exchange", "currency", "fx", "macroeconomic", "inflation",
    "dependence", "key customer", "key supplier", "threat", "threats",
    "exposure",
}

# Intent → (section_keywords, preferred_source_types)
_INTENT_SECTIONS: dict[str, tuple[list[str], list[str]]] = {
    "metric": (
        ["financial statements", "balance sheet", "md&a",
         "management's discussion", "results of operations",
         "cash flow", "reserved", "liquidity"],
        [],
    ),
    "risk": (
        ["risk factors", "legal proceedings", "quantitative"],
        [],
    ),
    "sentiment": (
        ["prepared remarks", "q&a", "transcript"],
        ["transcript"],
    ),
    "general": ([], []),
}


def classify_intent(query: str) -> str:
    """Return 'metric' | 'risk' | 'sentiment' | 'general' for rerank biasing."""
    q = query.lower()
    if any(s in q for s in _SENTIMENT_SIGNALS):
        logger.info("[HyDE] Intent: sentiment")
        return "sentiment"
    if any(s in q for s in _RISK_SIGNALS):
        logger.info("[HyDE] Intent: risk")
        return "risk"
    if any(s in q for s in _NUMERICAL_SIGNALS):
        logger.info("[HyDE] Intent: metric")
        return "metric"
    logger.info("[HyDE] Intent: general")
    return "general"


# ── Sub-query decomposition for numerical queries ────────────────────────────

# Maps high-level financial concepts → component keyword queries for BM25
_RATIO_DECOMPOSITION: dict[str, list[str]] = {
    "current ratio": [
        "current assets current liabilities balance sheet",
        "current assets total",
        "current liabilities total",
    ],
    "debt-to-equity": [
        "total liabilities stockholders equity balance sheet",
        "long-term debt shareholders equity",
    ],
    "debt to equity": [
        "total liabilities stockholders equity balance sheet",
        "long-term debt shareholders equity",
    ],
    "quick ratio": [
        "cash equivalents accounts receivable current liabilities",
        "liquid assets current liabilities",
    ],
    "gross margin": [
        "revenue cost of goods sold gross profit",
        "net sales cost of revenue",
    ],
    "operating margin": [
        "operating income revenue operating expenses",
        "income from operations net sales",
    ],
    "liquidity": [
        "current assets current liabilities working capital",
        "cash short-term investments liquid assets",
    ],
    "balance sheet": [
        "total assets total liabilities stockholders equity",
        "current assets current liabilities",
    ],
}


def _decompose_numerical_query(query: str) -> list[str]:
    """Return BM25-only sub-queries for the financial concept in the query."""
    q = query.lower()
    sub_queries: list[str] = []
    for concept, expansions in _RATIO_DECOMPOSITION.items():
        if concept in q:
            sub_queries.extend(expansions)
            logger.debug(f"[HyDE] Decomposed '{concept}' → {len(expansions)} sub-queries")
    # Always add bare component terms for any numerical query
    if not sub_queries:
        sub_queries = [
            "current assets current liabilities balance sheet",
            "total assets total liabilities equity",
        ]
    return sub_queries


# ── Prompt templates — GENERAL path ──────────────────────────────────────────

_GENERAL_QUERY_GEN_SYSTEM = """\
You are a senior financial analyst. Given a user question, generate exactly 4 \
alternative queries that each approach the topic from a DIFFERENT financial angle.

Perspectives to cover (one each):
1. QUANTITATIVE — focus on numbers, ratios, revenue, margins, balance-sheet items
2. RISK — focus on risk factors, threats, regulatory exposure, litigation, supply chain
3. TEMPORAL — focus on year-over-year trends, sequential changes, multi-period comparisons
4. STRATEGIC — focus on business segments, competitive position, management decisions, guidance

Rules:
- Do NOT paraphrase the original question — each query must have a fundamentally different focus.
- Write queries the way a financial analyst reading a 10-K or earnings transcript would ask them.
- Output ONLY a valid JSON array of exactly 4 strings, no markdown, no explanation.

Example for "What were Apple's main challenges in 2023?":
[
  "What were Apple's gross margin trends and revenue by product segment in fiscal 2023?",
  "What supply chain, geopolitical, and competitive risks did Apple disclose in its 2023 10-K?",
  "How did Apple's operating income and free cash flow change from fiscal 2022 to fiscal 2023?",
  "What forward-looking guidance and strategic priorities did Apple management emphasise in 2023 earnings calls?"
]"""

_GENERAL_HYDE_SYSTEM = """\
You are a financial analyst writing a passage that would appear in a SEC 10-K filing, \
10-Q report, or earnings call transcript.

Given a financial question, write a 150–250 word answer in the style of an actual \
financial document:
- Formal, precise prose (no bullet points, no markdown headers)
- Reference plausible financial metrics where natural (exact numbers optional)
- Structure: context → key factors / evidence → implication or conclusion
- Write as if this is an excerpt from a real filing, not a summary

Do NOT add disclaimers, introductory phrases like "Based on…", or meta-commentary.
Start directly with the financial content."""


# ── Prompt templates — NUMERICAL path ────────────────────────────────────────

_NUMERICAL_QUERY_GEN_SYSTEM = """\
You are a senior financial analyst writing search queries to retrieve data from \
SEC filings, earnings transcripts, and financial statements.

Given a numerical or ratio-based financial question, write exactly 4 search queries \
— one of each type below. Every query MUST be a complete, natural-language sentence \
or question, written the way an analyst would phrase it when searching a 10-K or \
earnings release.

Query types (one each):
1. DIRECT METRIC   — ask for the specific metric or ratio by name and period
2. FINANCIAL STATEMENT — ask for the relevant income statement or balance sheet figures
3. COMPONENT ITEMS — ask for the individual line items needed to compute the metric
4. CROSS-COMPANY   — ask how the metric compares across companies or to the industry

FORMAT RULES (strictly enforced):
- Each query must be a grammatically complete sentence ending with a period or question mark.
- Do NOT write keyword strings, formulas, or lists of terms.
- Do NOT include words like "keyword:", "query:", or structural labels in the output.
- Output ONLY a valid JSON array of 4 strings — no markdown, no explanation.

Example for "Which companies have the highest liquidity risk based on current ratio?":
[
  "What is the current ratio for each major company, and which ones carry the highest liquidity risk?",
  "How do current assets and current liabilities appear on the consolidated balance sheets of large-cap companies?",
  "What are the reported values of current assets and current liabilities for each company in their most recent fiscal year?",
  "Which companies have the lowest current ratio and therefore face the greatest short-term liquidity pressure?"
]

Example for "revenue growth Microsoft":
[
  "What was Microsoft's year-over-year revenue growth rate in the most recent fiscal year?",
  "How did Microsoft's total revenue change between fiscal 2023 and fiscal 2024 as reported in the income statement?",
  "What were the individual product and segment revenue line items driving Microsoft's overall revenue trend?",
  "How does Microsoft's revenue growth rate compare to peers such as Alphabet, Apple, and Amazon?"
]"""

_NUMERICAL_QUERY_GEN_USER = "Financial question: {query}\n\nGenerate 4 natural-language search queries (JSON array only):"

_NUMERICAL_HYDE_SYSTEM = """\
You are a financial analyst. Write a 4–6 sentence passage describing the relevant \
financial data for the question below. The passage should read like an excerpt from \
a 10-K filing note, earnings release, or analyst report.

Requirements:
- Write in complete, formal sentences (not bullet points or tables).
- Include specific financial line items relevant to the question, e.g. "current assets", \
  "current liabilities", "total revenue", "net income", "shareholders equity".
- Include plausible approximate dollar figures (e.g. "approximately $143 billion" or \
  "roughly $29.4 billion") — exact accuracy is not required.
- If the question is comparative, mention 2–3 companies with their respective figures.
- Compute or reference the target metric explicitly (e.g. "resulting in a current ratio \
  of approximately 1.2").

Do not add disclaimers, introductions ("Based on…"), or labels ("Answer:").
Begin the passage immediately with the financial content.

Example:
"As of the most recent fiscal year-end, the company reported current assets of \
approximately $143 billion and current liabilities of approximately $134 billion, \
yielding a current ratio of roughly 1.07. Current assets included cash and cash \
equivalents of $30 billion, short-term investments of $32 billion, and accounts \
receivable of $28 billion. Current liabilities were primarily composed of accounts \
payable of $63 billion and accrued expenses of $45 billion. This current ratio \
indicates adequate but tight short-term liquidity relative to peers with ratios \
above 1.5.\""""

_NUMERICAL_HYDE_USER = "Financial question: {query}\n\nWrite the financial data passage:"

# Shared user template for general path
_GENERAL_QUERY_GEN_USER = "Original question: {query}\n\nGenerate 4 diverse alternative queries:"
_GENERAL_HYDE_USER = "Question: {query}\n\nWrite the financial document passage:"


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _is_keyword_stuffed(text: str) -> bool:
    """Return True if text looks like a keyword string rather than a natural sentence."""
    words = text.split()
    if len(words) < 6:
        return True
    if not text.rstrip().endswith((".", "?", "!")):
        keyword_fraction = sum(1 for w in words if w.islower() and len(w) > 3) / len(words)
        if keyword_fraction > 0.85:
            return True
    return False


def _fallback_queries(query: str, query_type: QueryType) -> list[str]:
    """Return 4 natural-language template queries without an LLM call."""
    q = query.strip().rstrip("?")
    if query_type == QueryType.NUMERICAL:
        return [
            f"What are the reported figures for {q} in the most recent fiscal year?",
            f"How does {q} appear in the consolidated financial statements?",
            f"What individual line items from the balance sheet or income statement relate to {q}?",
            f"How does {q} compare across major companies in the sector?",
        ]
    return [
        f"What do SEC filings and earnings reports say about {q}?",
        f"What risk factors and management disclosures relate to {q}?",
        f"How has {q} trended over recent fiscal years?",
        f"What is management's strategy and forward guidance regarding {q}?",
    ]


def _fallback_hypothesis(query: str, query_type: QueryType) -> str:
    """Return a minimal but valid template hypothesis when LLM generation fails."""
    q = query.strip().rstrip("?")
    if query_type == QueryType.NUMERICAL:
        return (
            f"The company disclosed financial data directly relevant to {q}. "
            f"As reported in the consolidated balance sheet, current assets totalled "
            f"approximately $120 billion and current liabilities were approximately $95 billion, "
            f"resulting in a current ratio of roughly 1.26. "
            f"Total assets were approximately $340 billion and total liabilities "
            f"approximately $270 billion, with shareholders equity of approximately $70 billion. "
            f"These figures are drawn from the most recent annual report filed with the SEC."
        )
    return (
        f"According to the company's most recent SEC filings and earnings disclosures, "
        f"{q} represents a significant area of focus for management. "
        f"In the annual report's Management Discussion and Analysis section, "
        f"the company addressed this topic in detail, noting both opportunities "
        f"and risks relevant to its business operations and financial performance. "
        f"Management's forward-looking statements indicated continued attention "
        f"to this area in the coming fiscal year."
    )


def generate_diverse_queries(
    query: str,
    client: "OpenAI",
    model: str,
    query_type: QueryType = QueryType.GENERAL,
) -> list[str]:
    """Generate 4 diverse, natural-language financial queries from the original question.

    Uses type-aware prompts. Falls back to template queries — never returns keyword strings.
    """
    if query_type == QueryType.NUMERICAL:
        system = _NUMERICAL_QUERY_GEN_SYSTEM
        user   = _NUMERICAL_QUERY_GEN_USER.format(query=query)
    else:
        system = _GENERAL_QUERY_GEN_SYSTEM
        user   = _GENERAL_QUERY_GEN_USER.format(query=query)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_completion_tokens=3000,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("LLM returned empty content (token budget consumed by reasoning)")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        queries: list[str] = json.loads(raw)
        if not isinstance(queries, list) or not queries:
            raise ValueError("Empty or non-list JSON response")

        queries = [str(q).strip() for q in queries[:5] if str(q).strip()]

        # Replace any keyword-stuffed outputs with template fallbacks
        cleaned: list[str] = []
        for q in queries:
            if _is_keyword_stuffed(q):
                logger.warning(f"[HyDE] Keyword-stuffed query rejected: '{q[:80]}'")
            else:
                cleaned.append(q)

        if not cleaned:
            raise ValueError("All generated queries were keyword-stuffed")

        label = "numerical" if query_type == QueryType.NUMERICAL else "general"
        logger.info(f"[HyDE] Generated {len(cleaned)} {label} queries")
        for i, q in enumerate(cleaned, 1):
            logger.info(f"[HyDE]   Q{i}: {q}")
        return cleaned

    except Exception as e:
        logger.warning(f"[HyDE] Query generation failed ({e}) — using template fallback queries")
        fallback = _fallback_queries(query, query_type)
        for i, q in enumerate(fallback, 1):
            logger.info(f"[HyDE]   Q{i} [fallback]: {q}")
        return fallback


def generate_hypothetical_doc(
    query: str,
    client: "OpenAI",
    model: str,
    query_type: QueryType = QueryType.GENERAL,
) -> str:
    """Generate a hypothetical document passage that answers the query.

    ALWAYS returns a non-empty string — falls back to a template passage if the
    LLM call fails or returns empty content (e.g. token budget exhausted by reasoning).
    """
    if query_type == QueryType.NUMERICAL:
        system = _NUMERICAL_HYDE_SYSTEM
        user   = _NUMERICAL_HYDE_USER.format(query=query)
    else:
        system = _GENERAL_HYDE_SYSTEM
        user   = _GENERAL_HYDE_USER.format(query=query)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            # 4000 tokens: reasoning model uses ~1000-2000 for internal reasoning,
            # leaving 2000-3000 for visible output (~300 words).
            max_completion_tokens=4000,
        )
        doc = (response.choices[0].message.content or "").strip()
        if not doc:
            raise ValueError("LLM returned empty content (all tokens consumed by reasoning)")
        logger.debug(f"[HyDE] Hypothesis ({len(doc)} chars): {doc[:120]}…")
        return doc

    except Exception as e:
        logger.warning(
            f"[HyDE] Hypothesis generation failed for '{query[:60]}' ({e}) "
            f"— using template fallback"
        )
        return _fallback_hypothesis(query, query_type)


# ── Weighted multi-list RRF ───────────────────────────────────────────────────

def _multi_list_rrf(
    dense_lists: list[list[dict]],
    sparse_lists: list[list[tuple[int, float]]],
    faiss_chunks: list[dict],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion across multiple dense and sparse lists.

    For each document d:
        rrf_score(d) = dense_weight  × Σ_dense  1/(rank + k)
                     + sparse_weight × Σ_sparse 1/(rank + k)

    Args:
        dense_lists:   FAISS result lists (chunk dicts with 'score').
        sparse_lists:  BM25 result lists [(global_idx, bm25_score)].
        faiss_chunks:  Full chunk store for global_idx → chunk dict lookup.
        k:             RRF constant (60 is standard).
        dense_weight:  Multiplier for dense retrieval contributions.
                       For GENERAL queries: 0.5. For NUMERICAL: 0.4.
        sparse_weight: Multiplier for sparse retrieval contributions.
                       For GENERAL queries: 0.5. For NUMERICAL: 0.6.

    Returns:
        Merged, deduplicated chunk dicts sorted by rrf_score descending.
    """
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    # Dense contributions (weighted)
    for ranked_list in dense_lists:
        for rank, chunk in enumerate(ranked_list, 1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (rank + k)
            chunk_by_id[cid] = chunk

    # Sparse contributions (weighted)
    for sparse_list in sparse_lists:
        for rank, (global_idx, _) in enumerate(sparse_list, 1):
            if global_idx >= len(faiss_chunks):
                continue
            chunk = faiss_chunks[global_idx]
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight / (rank + k)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = {**chunk, "score": 0.0}

    result: list[dict] = []
    for cid in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True):
        chunk = chunk_by_id[cid].copy()
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        result.append(chunk)

    return result


# ── MultiHyDE Retriever ───────────────────────────────────────────────────────

class MultiHyDERetriever:
    """Pre-retrieval transformation layer: diverse queries → hypothetical docs → retrieval.

    Automatically switches between GENERAL and NUMERICAL retrieval strategies
    based on keyword detection in the incoming query.  NUMERICAL queries get:
    - Metric/component/table/comparison query generation (no narrative queries)
    - Balance-sheet-style hypothetical documents with explicit line items + numbers
    - Increased BM25 weight (sparse=0.6) for exact financial term matching
    - Automatic sub-query decomposition into component keyword searches
    """

    # Retrieval weights per query type
    _WEIGHTS = {
        QueryType.GENERAL:   {"dense": 0.5, "sparse": 0.5},
        QueryType.NUMERICAL: {"dense": 0.4, "sparse": 0.6},
    }

    def __init__(
        self,
        faiss_manager: "FAISSManager",
        bm25_index: "BM25Index",
        embedder: "TitanEmbedder",
        llm_client: "OpenAI",
        llm_model: str,
        num_hypotheses: int = 4,
    ):
        self.faiss = faiss_manager
        self.bm25 = bm25_index
        self.embedder = embedder
        self.llm = llm_client
        self.model = llm_model
        self.num_hypotheses = num_hypotheses

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        ticker_filter: str | None = None,
        dense_k: int = 40,
        sparse_k: int = 40,
        rrf_k: int = 60,
        rerank_input: int = 30,
        final_k: int = 24,
    ) -> list[dict]:
        """Run Multi-HyDE retrieval with automatic query-type adaptation.

        Args:
            query:         Original user query.
            ticker_filter: Optional ticker to restrict retrieval to one company.
            dense_k:       FAISS results per hypothesis vector.
            sparse_k:      BM25 results per diverse / sub-query.
            rrf_k:         RRF constant.
            rerank_input:  Candidates passed to BGE reranker.
            final_k:       Chunks returned.

        Returns:
            Reranked list of chunk dicts, each with 'score' = sigmoid(bge_logit)
            and 'rrf_score' for traceability.
        """
        # ── Step 0: Classify query type ───────────────────────────────────────
        qtype = classify_query(query)
        weights = self._WEIGHTS[qtype]

        # ── Step 1: Generate diverse queries (type-aware) ─────────────────────
        diverse_queries = generate_diverse_queries(
            query, self.llm, self.model, query_type=qtype
        )
        diverse_queries = diverse_queries[: self.num_hypotheses]

        # ── Step 2: Generate hypothetical documents in parallel ───────────────
        hyp_docs = self._generate_hypotheses_parallel(diverse_queries, qtype)
        if not hyp_docs:
            logger.warning("[HyDE] All hypotheses failed — falling back to original query embed")
            hyp_docs = [query]

        logger.info(f"[HyDE] {len(hyp_docs)} hypothetical docs ready for embedding")

        # ── Step 3: Embed hypothetical documents ──────────────────────────────
        vecs: np.ndarray = self.embedder.embed_batch(hyp_docs, show_progress=False)

        # ── Step 4: Ticker candidate IDs for BM25 filtering ──────────────────
        candidate_ids: set[int] | None = (
            set(self.faiss.metadata_index["by_ticker"].get(ticker_filter, []))
            if ticker_filter else None
        )

        # ── Step 5: Dense retrieval — one FAISS search per hypothesis vector ──
        dense_lists: list[list[dict]] = []
        for i, vec in enumerate(vecs):
            hits = self.faiss.search(vec, k=dense_k, ticker=ticker_filter)
            dense_lists.append(hits)
            logger.debug(f"[HyDE] Dense[{i}]: {len(hits)} hits")

        # ── Step 6: Sparse retrieval — diverse queries + sub-queries ─────────
        # For NUMERICAL: also search decomposed component terms so exact
        # financial keywords ("current assets", "current liabilities") can
        # surface balance-sheet chunks that semantic search might miss.
        bm25_queries = list(diverse_queries)
        if qtype == QueryType.NUMERICAL:
            sub_queries = _decompose_numerical_query(query)
            bm25_queries.extend(sub_queries)
            logger.info(f"[HyDE] Added {len(sub_queries)} decomposed sub-queries for BM25")

        sparse_lists: list[list[tuple[int, float]]] = []
        for i, bq in enumerate(bm25_queries):
            hits = self.bm25.search(bq, k=sparse_k, candidate_ids=candidate_ids)
            sparse_lists.append(hits)
            logger.debug(f"[HyDE] Sparse[{i}] '{bq[:50]}': {len(hits)} hits")

        # ── Step 7: Weighted multi-list RRF fusion ────────────────────────────
        fused = _multi_list_rrf(
            dense_lists, sparse_lists, self.faiss.chunks,
            k=rrf_k,
            dense_weight=weights["dense"],
            sparse_weight=weights["sparse"],
        )
        candidates = fused[:rerank_input]
        logger.info(
            f"[HyDE] RRF ({qtype.value}, dense={weights['dense']}, sparse={weights['sparse']}): "
            f"{len(fused)} unique → top {len(candidates)} to reranker"
        )

        # ── Step 8: BGE reranking with the ORIGINAL query ─────────────────────
        intent = classify_intent(query)
        section_kws, pref_sources = _INTENT_SECTIONS[intent]
        reranked = _rerank(
            query,
            candidates,
            top_k=final_k,
            section_keywords=section_kws or None,
            preferred_source_types=pref_sources or None,
        )
        if reranked:
            logger.info(
                f"[HyDE] Reranked {len(reranked)} chunks, "
                f"top score={reranked[0]['score']:.3f}"
            )
        else:
            logger.warning("[HyDE] 0 chunks after rerank")

        return reranked

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_hypotheses_parallel(
        self,
        queries: list[str],
        query_type: QueryType,
    ) -> list[str]:
        """Generate hypothetical documents for each query in parallel threads."""
        results: list[str] = [""] * len(queries)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futures = {
                pool.submit(
                    generate_hypothetical_doc, q, self.llm, self.model, query_type
                ): i
                for i, q in enumerate(queries)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"[HyDE] Hypothesis thread {idx} raised: {e}")

        non_empty = [d for d in results if d.strip()]
        logger.info(f"[HyDE] {len(non_empty)}/{len(queries)} hypotheses generated successfully")
        return non_empty
