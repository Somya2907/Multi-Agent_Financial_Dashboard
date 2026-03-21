"""Retrieval agent: embed query → search FAISS → filter noise → aggregate context.

Wraps the existing embedding, indexing, and reasoning sub-modules so the
LangGraph pipeline can call them as a single node.
"""

import logging

from pipeline.embedding.embedder import TitanEmbedder
from pipeline.indexing.faiss_manager import FAISSManager
from pipeline.reasoning.noise_filter import filter_noise
from pipeline.reasoning.aggregator import aggregate_chunks

logger = logging.getLogger(__name__)

_RETRIEVAL_K = 20
_MIN_SCORE = 0.45
_MAX_CHUNKS = 12

# Module-level singletons — loaded once per process
_embedder: TitanEmbedder | None = None
_index: FAISSManager | None = None


def _load_if_needed():
    global _embedder, _index
    if _embedder is None:
        _embedder = TitanEmbedder()
    if _index is None:
        _index = FAISSManager()
        _index.load()


def run_retrieval(state: dict) -> dict:
    """Retrieve and filter relevant chunks for the query.

    Uses a single-ticker FAISS filter when query_type == "single_company"
    and a known ticker is available.  For comparison and general queries,
    no ticker filter is applied so chunks from all companies can surface.

    Returns a partial state update with:
        retrieved_chunks, context, citations
    """
    _load_if_needed()

    query = state["query"]
    query_type = state.get("query_type", "general")
    tickers = state.get("tickers", [])

    # Apply ticker filter only for single-company queries
    ticker_filter = tickers[0] if query_type == "single_company" and len(tickers) == 1 else None

    logger.info(
        f"[Retrieval] Searching (query_type={query_type}, "
        f"ticker_filter={ticker_filter}, k={_RETRIEVAL_K})"
    )

    query_vec = _embedder.embed_single(query)
    raw_chunks = _index.search(query_vec, k=_RETRIEVAL_K, ticker=ticker_filter)
    logger.info(f"[Retrieval] {len(raw_chunks)} raw chunks retrieved")

    filtered = filter_noise(raw_chunks, min_score=_MIN_SCORE, max_chunks=_MAX_CHUNKS)
    logger.info(f"[Retrieval] {len(filtered)} chunks after noise filter")

    if not filtered:
        return {
            "retrieved_chunks": [],
            "context": "",
            "citations": [],
        }

    context, citations = aggregate_chunks(filtered)
    return {
        "retrieved_chunks": filtered,
        "context": context,
        "citations": citations,
    }
