"""Post-RAG Reasoning Layer: retrieve → filter → aggregate → metrics → answer."""

import logging
from dataclasses import dataclass, field

from pipeline.embedding.embedder import TitanEmbedder
from pipeline.indexing.faiss_manager import FAISSManager
from pipeline.reasoning.noise_filter import filter_noise
from pipeline.reasoning.aggregator import aggregate_chunks
from pipeline.reasoning.metrics_computer import compute_metrics
from pipeline.reasoning.answer_generator import generate_answer

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Structured output from the reasoning layer."""
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    retrieved_chunks: int = 0
    chunks_after_filter: int = 0


class QueryEngine:
    """End-to-end reasoning pipeline over the FAISS index.

    Steps for each query:
      1. Embed query via Amazon Titan
      2. Retrieve top-k chunks from FAISS (with optional metadata filters)
      3. Filter noise (score threshold + deduplication)
      4. Aggregate chunks into structured context
      5. Compute financial metrics from cached structured data
      6. Generate final answer via GPT-5
    """

    def __init__(
        self,
        retrieval_k: int = 20,
        min_score: float = 0.45,
        max_context_chunks: int = 12,
    ):
        self.retrieval_k = retrieval_k
        self.min_score = min_score
        self.max_context_chunks = max_context_chunks

        self._embedder: TitanEmbedder | None = None
        self._index: FAISSManager | None = None

    def _load_if_needed(self):
        if self._embedder is None:
            self._embedder = TitanEmbedder()
        if self._index is None:
            self._index = FAISSManager()
            self._index.load()

    def answer(
        self,
        query: str,
        ticker: str | None = None,
        source_type: str | None = None,
        section_name: str | None = None,
        include_metrics: bool = True,
    ) -> AnswerResult:
        """Run the full post-RAG reasoning pipeline for a query.

        Args:
            query: Natural language question.
            ticker: Restrict retrieval to this company (e.g., "AAPL").
            source_type: Restrict to "10-K", "10-Q", "transcript", or "news".
            section_name: Restrict to a specific section (e.g., "Risk Factors").
            include_metrics: If True and a ticker is provided, compute and inject
                             structured financial metrics into the LLM prompt.

        Returns:
            AnswerResult with answer, citations, and metrics.
        """
        self._load_if_needed()

        # ── Step 1: Embed query ──────────────────────────────────────────────
        logger.info(f"[1/5] Embedding query: {query!r}")
        query_vec = self._embedder.embed_single(query)

        # ── Step 2: Retrieve top-k chunks ────────────────────────────────────
        logger.info(f"[2/5] Retrieving top-{self.retrieval_k} chunks")
        raw_chunks = self._index.search(
            query_vec,
            k=self.retrieval_k,
            ticker=ticker,
            source_type=source_type,
            section_name=section_name,
        )
        logger.info(f"      Retrieved {len(raw_chunks)} chunks")

        # ── Step 3: Filter noise ─────────────────────────────────────────────
        logger.info("[3/5] Filtering noise")
        filtered = filter_noise(
            raw_chunks,
            min_score=self.min_score,
            max_chunks=self.max_context_chunks,
        )
        logger.info(f"      {len(raw_chunks)} → {len(filtered)} chunks after filter")

        if not filtered:
            return AnswerResult(
                query=query,
                answer="No sufficiently relevant passages were found for this query.",
                retrieved_chunks=len(raw_chunks),
                chunks_after_filter=0,
            )

        # ── Step 4: Aggregate chunks into context ────────────────────────────
        logger.info("[4/5] Aggregating context")
        context, citations = aggregate_chunks(filtered)

        # ── Step 5: Compute financial metrics ────────────────────────────────
        metrics: dict = {}
        if include_metrics and ticker:
            logger.info(f"[5/5] Computing metrics for {ticker}")
            metrics = compute_metrics(ticker)
        elif include_metrics and not ticker:
            # Try to infer tickers from retrieved chunks
            tickers_in_results = list(
                {c.get("ticker") for c in filtered if c.get("ticker")}
            )
            if len(tickers_in_results) == 1:
                logger.info(f"[5/5] Computing metrics for inferred ticker {tickers_in_results[0]}")
                metrics = compute_metrics(tickers_in_results[0])
            else:
                logger.info("[5/5] Skipping metrics (multi-company query)")

        # ── Step 6: Generate answer ──────────────────────────────────────────
        logger.info("[6/6] Generating answer")
        answer_text = generate_answer(query, context, metrics)

        return AnswerResult(
            query=query,
            answer=answer_text,
            citations=citations,
            metrics=metrics,
            retrieved_chunks=len(raw_chunks),
            chunks_after_filter=len(filtered),
        )
