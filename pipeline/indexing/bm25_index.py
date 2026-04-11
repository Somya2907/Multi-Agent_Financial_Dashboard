"""BM25 sparse index over the chunk store."""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indexes"


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


class BM25Index:
    """BM25 sparse retrieval over the chunk store.

    Row order in the BM25 index mirrors the chunk list passed to build(),
    which is the same order as the FAISS index — so BM25 row idx == FAISS global idx.
    """

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or INDEX_DIR
        self.bm25: BM25Okapi | None = None
        self.n_chunks: int = 0

    def build(self, chunks: list[dict]):
        """Build BM25 index from a list of chunk dicts (same order as FAISS)."""
        tokenized = [_tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.n_chunks = len(chunks)
        logger.info(f"[BM25] Built index: {self.n_chunks} chunks")

    def save(self, path: Path | None = None):
        path = path or (self.index_dir / "bm25_index.pkl")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "n_chunks": self.n_chunks}, f)
        logger.info(f"[BM25] Saved index to {path}")

    def load(self, path: Path | None = None):
        path = path or (self.index_dir / "bm25_index.pkl")
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index at {path}. Run the pipeline first.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.n_chunks = data["n_chunks"]
        logger.info(f"[BM25] Loaded index: {self.n_chunks} chunks")

    def search(
        self,
        query: str,
        k: int = 40,
        candidate_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return top-k (global_chunk_idx, normalized_score) pairs.

        Args:
            query: Raw query string.
            k: Number of results.
            candidate_ids: Optional set of global chunk indices to restrict search
                           (used for ticker filtering — mirrors FAISS metadata filter).

        Returns:
            List of (chunk_idx, score) sorted by score descending.
            Scores are normalized to [0, 1] relative to the top result.
        """
        if self.bm25 is None:
            raise RuntimeError("[BM25] Index not loaded — call build() or load() first")

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores: np.ndarray = self.bm25.get_scores(tokens)  # shape (n_chunks,)

        if candidate_ids is not None:
            mask = np.zeros(len(scores), dtype=bool)
            for idx in candidate_ids:
                if 0 <= idx < len(scores):
                    mask[idx] = True
            scores = np.where(mask, scores, 0.0)

        max_score = float(scores.max())
        if max_score <= 0:
            return []
        scores = scores / max_score  # normalize to [0, 1]

        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            (int(i), float(scores[i]))
            for i in top_k_idx
            if scores[i] > 0
        ]
