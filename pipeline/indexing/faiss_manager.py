"""FAISS index management with metadata filtering for two-stage retrieval."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "indexes"


class FAISSManager:
    """Manages a FAISS index with aligned chunk store and metadata inverted index."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []  # Aligned with FAISS IDs
        self.metadata_index: dict[str, dict[str, list[int]]] = {
            "by_ticker": defaultdict(list),
            "by_source_type": defaultdict(list),
            "by_fiscal_period": defaultdict(list),
            "by_section": defaultdict(list),
        }

    def build_index(self, vectors: np.ndarray, chunks: list[dict]):
        """Build a new FAISS index from vectors and chunk metadata.

        Args:
            vectors: (N, dim) float32 array of normalized embeddings.
            chunks: List of N chunk dicts (must be aligned with vectors).
        """
        n, dim = vectors.shape
        assert n == len(chunks), f"Vector count ({n}) != chunk count ({len(chunks)})"

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.chunks = chunks

        # Build metadata inverted index
        self.metadata_index = {
            "by_ticker": defaultdict(list),
            "by_source_type": defaultdict(list),
            "by_fiscal_period": defaultdict(list),
            "by_section": defaultdict(list),
        }
        for i, chunk in enumerate(chunks):
            self.metadata_index["by_ticker"][chunk.get("ticker", "")].append(i)
            self.metadata_index["by_source_type"][chunk.get("source_type", "")].append(i)
            self.metadata_index["by_fiscal_period"][chunk.get("fiscal_period", "")].append(i)
            self.metadata_index["by_section"][chunk.get("section_name", "")].append(i)

        logger.info(f"Built FAISS index: {n} vectors, dim={dim}")

    def save(self):
        """Save index, chunk store, and metadata to disk."""
        faiss.write_index(self.index, str(self.index_dir / "faiss_index.bin"))

        with open(self.index_dir / "chunk_store.jsonl", "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")

        # Convert defaultdicts to regular dicts for JSON serialization
        meta = {
            k: dict(v) for k, v in self.metadata_index.items()
        }
        with open(self.index_dir / "metadata_index.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved index to {self.index_dir}")

    def load(self):
        """Load index, chunk store, and metadata from disk."""
        index_path = self.index_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(str(index_path))

        self.chunks = []
        with open(self.index_dir / "chunk_store.jsonl") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        with open(self.index_dir / "metadata_index.json") as f:
            raw = json.load(f)
            self.metadata_index = {
                k: defaultdict(list, v) for k, v in raw.items()
            }

        logger.info(
            f"Loaded FAISS index: {self.index.ntotal} vectors, "
            f"{len(self.chunks)} chunks"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ticker: str | None = None,
        source_type: str | None = None,
        section_name: str | None = None,
    ) -> list[dict]:
        """Search the index with optional metadata filtering.

        Args:
            query_vector: (dim,) normalized query embedding.
            k: Number of results to return.
            ticker: Filter to specific company.
            source_type: Filter to "10-K", "10-Q", "transcript", "news".
            section_name: Filter to specific section.

        Returns:
            List of chunk dicts with added 'score' field, sorted by relevance.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index() or load() first.")

        # Compute candidate IDs from metadata filters
        candidate_ids = None

        filters = [
            ("by_ticker", ticker),
            ("by_source_type", source_type),
            ("by_section", section_name),
        ]
        for meta_key, filter_value in filters:
            if filter_value:
                ids = set(self.metadata_index[meta_key].get(filter_value, []))
                candidate_ids = ids if candidate_ids is None else candidate_ids & ids

        # If we have filter candidates, use IDSelector for filtered search
        if candidate_ids is not None:
            if not candidate_ids:
                return []
            id_array = np.array(sorted(candidate_ids), dtype=np.int64)
            selector = faiss.IDSelectorBatch(id_array)
            params = faiss.SearchParametersIVF()
            params.sel = selector
            # For FlatIP, we need to use search with IDSelector via a wrapper
            # Use a simpler approach: search broader then filter
            search_k = min(k * 10, self.index.ntotal)
            query_2d = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_2d, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if int(idx) in candidate_ids:
                    chunk = self.chunks[int(idx)].copy()
                    chunk["score"] = float(score)
                    results.append(chunk)
                    if len(results) >= k:
                        break
            return results
        else:
            # Unfiltered search
            query_2d = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_2d, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk = self.chunks[int(idx)].copy()
                chunk["score"] = float(score)
                results.append(chunk)
            return results
