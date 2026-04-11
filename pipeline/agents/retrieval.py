"""Retrieval agent: Multi-HyDE pre-retrieval → hybrid BM25 + Titan dense → BGE reranker.

Flow
----
    query
      │
      ▼  MultiHyDERetriever.run()
      │    ├─ LLM: generate 4 diverse financial queries
      │    ├─ LLM (parallel): hypothetical SEC doc per query
      │    ├─ Titan: embed_batch(hyp_docs)          ← NEVER embeds original query
      │    ├─ 4 × FAISS dense search (one per vec)
      │    ├─ 4 × BM25 sparse search (one per diverse query)
      │    ├─ multi-list RRF fusion
      │    └─ BGE rerank with ORIGINAL query
      │
      ▼  filter_noise (dedup same-section, cap at _MAX_CHUNKS)
      ▼  aggregate_chunks → context string + citations
"""

# KMP_DUPLICATE_LIB_OK must be set before any library initializes its OpenMP
# runtime.  On macOS, FAISS (libomp) and PyTorch (libomp) both ship their own
# copy; without this flag the process segfaults when both are loaded.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging

# Import sentence-transformers (PyTorch) BEFORE faiss so torch's OpenMP
# runtime is the first one to initialize — prevents macOS segfault.
from openai import OpenAI

from config.settings import settings
from pipeline.embedding.embedder import TitanEmbedder
from pipeline.indexing.faiss_manager import FAISSManager
from pipeline.indexing.bm25_index import BM25Index
from pipeline.reasoning.hyde import MultiHyDERetriever
from pipeline.reasoning.noise_filter import filter_noise
from pipeline.reasoning.aggregator import aggregate_chunks

logger = logging.getLogger(__name__)

_MAX_CHUNKS = 12     # Final output cap (after dedup)

# Module-level singletons — loaded once per process
_embedder: TitanEmbedder | None = None
_faiss: FAISSManager | None = None
_bm25: BM25Index | None = None
_llm: OpenAI | None = None
_hyde: MultiHyDERetriever | None = None


def _load_if_needed():
    global _embedder, _faiss, _bm25, _llm, _hyde
    if _embedder is None:
        _embedder = TitanEmbedder()
    if _faiss is None:
        _faiss = FAISSManager()
        _faiss.load()
    if _bm25 is None:
        _bm25 = BM25Index()
        _bm25.load()
    if _llm is None:
        _llm = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    if _hyde is None:
        _hyde = MultiHyDERetriever(
            faiss_manager=_faiss,
            bm25_index=_bm25,
            embedder=_embedder,
            llm_client=_llm,
            llm_model=settings.openai_llm_model,
            num_hypotheses=4,
        )


def run_retrieval(state: dict) -> dict:
    """Multi-HyDE hybrid retrieval with BGE reranking.

    Identical external signature to the previous single-query retrieval agent.
    Internally replaces the single query-embedding step with:
        diverse queries → hypothetical docs → embeddings → multi-list RRF → rerank

    Returns a partial state update with: retrieved_chunks, context, citations.
    """
    _load_if_needed()

    query = state["query"]
    query_type = state.get("query_type", "general")
    tickers = state.get("tickers", [])

    # Apply ticker filter only for single-company queries
    ticker_filter = (
        tickers[0] if query_type == "single_company" and len(tickers) == 1 else None
    )

    logger.info(
        f"[Retrieval] Multi-HyDE — query_type={query_type}, "
        f"ticker_filter={ticker_filter}"
    )

    # ── Multi-HyDE: diverse queries → hypotheses → multi-list FAISS+BM25 → rerank ──
    reranked = _hyde.run(
        query=query,
        ticker_filter=ticker_filter,
        dense_k=40,
        sparse_k=40,
        rrf_k=60,
        rerank_input=30,
        final_k=_MAX_CHUNKS * 2,  # give filter_noise room to dedup
    )

    # ── Dedup same-section chunks and cap ──────────────────────────────────────
    # min_score=0.0: BGE reranker is the quality gate; we only dedup + cap here.
    filtered = filter_noise(reranked, min_score=0.0, max_chunks=_MAX_CHUNKS)
    logger.info(f"[Retrieval] Final: {len(filtered)} chunks after dedup/cap")

    if not filtered:
        return {"retrieved_chunks": [], "context": "", "citations": []}

    context, citations = aggregate_chunks(filtered)
    return {
        "retrieved_chunks": filtered,
        "context": context,
        "citations": citations,
    }
