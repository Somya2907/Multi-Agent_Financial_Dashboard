"""BGE cross-encoder reranker using sentence-transformers.

OpenMP safety on macOS
----------------------
FAISS, PyTorch, and Bedrock's HTTP connection pool each manage their own thread
pools.  On macOS ARM, having multiple OMP runtimes active simultaneously causes a
segfault during BGE inference.  Three mitigations are applied here:

1. OMP_NUM_THREADS / MKL_NUM_THREADS = 1  — disables multi-threading in all OMP
   runtimes so they cannot conflict regardless of initialization order.
2. KMP_DUPLICATE_LIB_OK = TRUE             — suppress the hard-abort on duplicate
   libomp detection (safety net).
3. Module-level CrossEncoder import        — ensures torch initializes before faiss
   in any process that imports this module.
"""

import logging
import os

# Must be set before torch/OMP initializes — force single-threaded mode
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
# Module-level import ensures torch initializes its OpenMP runtime before faiss.
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-base"

# Pre-warm at import time so PyTorch's OMP thread pool is initialized before
# any other threaded work (e.g., parallel LLM calls in MultiHyDE) takes place.
# This prevents a segfault caused by OMP thread-pool conflicts on macOS when
# concurrent.futures threads are still alive during first BGE inference.
logger.info(f"[Reranker] Loading {_MODEL_NAME}...")
_reranker: CrossEncoder = CrossEncoder(_MODEL_NAME, device="cpu")
logger.info("[Reranker] Model loaded")


def _load_reranker() -> CrossEncoder:
    return _reranker


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def rerank(query: str, chunks: list[dict], top_k: int = 24) -> list[dict]:
    """Rerank chunks using the BGE cross-encoder.

    Args:
        query: User query string.
        chunks: Candidate chunks, each with a 'text' field.
        top_k: Number of chunks to return after reranking.

    Returns:
        Top-k chunks sorted by cross-encoder score descending.
        Each chunk's 'score' field is overwritten with sigmoid(logit),
        so scores remain in [0, 1] and are compatible with downstream
        noise_filter thresholds.
    """
    if not chunks:
        return []

    model = _load_reranker()
    pairs = [(query, c["text"]) for c in chunks]

    try:
        with torch.inference_mode():
            logits = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"[Reranker] Prediction failed: {e} — returning original order")
        return chunks[:top_k]

    scored = [
        {**chunk, "score": _sigmoid(float(logit))}
        for chunk, logit in zip(chunks, logits)
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)

    logger.info(
        f"[Reranker] Top scores: "
        + ", ".join(f"{c['score']:.3f}" for c in scored[:5])
    )
    return scored[:top_k]
