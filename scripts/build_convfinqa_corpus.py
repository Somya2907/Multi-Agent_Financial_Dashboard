"""Build a retrieval corpus from ConvFinQA's own passages.

ConvFinQA ships each example with pre_text + table + post_text drawn from a 10-K
page. Deduplicating by `filename` across train+dev gives ~1.8k unique passages.
We embed each one with Titan V2 and build a FAISS + BM25 index so we can run
our Multi-HyDE retriever over ConvFinQA's corpus — a clean IR-style eval that
tests the retrieval *algorithm* independent of our 50-ticker domain index.

Output:
    data/indexes/convfinqa/faiss_index.bin
    data/indexes/convfinqa/chunk_store.jsonl
    data/indexes/convfinqa/metadata_index.json
    data/indexes/convfinqa/bm25_index.pkl

Usage:
    python scripts/build_convfinqa_corpus.py
    python scripts/build_convfinqa_corpus.py --limit 50   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.embedding.embedder import TitanEmbedder  # noqa: E402
from pipeline.indexing.faiss_manager import FAISSManager  # noqa: E402
from pipeline.indexing.bm25_index import BM25Index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("build_convfinqa_corpus")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "convfinqa" / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes" / "convfinqa"

# filename format: "{TICKER}/{YEAR}/page_{N}.pdf"
_FILENAME_RE = re.compile(r"^([A-Z.\-]+)/(\d{4})/page_(\d+)\.pdf$")


def _serialize_table(rows: list[list[str]]) -> str:
    """Render a table as a pipe-separated text block."""
    if not rows:
        return ""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in rows)


def _build_chunk_text(ex: dict) -> str:
    pre = " ".join(ex.get("pre_text", []) or [])
    post = " ".join(ex.get("post_text", []) or [])
    table = _serialize_table(ex.get("table", []) or [])
    return f"{pre}\n\nTABLE:\n{table}\n\n{post}".strip()


def _parse_filename(fn: str) -> tuple[str, str, str]:
    """Return (ticker, year, page) parsed from 'TICKER/YEAR/page_N.pdf'."""
    m = _FILENAME_RE.match(fn)
    if not m:
        return ("UNK", "UNK", "0")
    return m.group(1), m.group(2), m.group(3)


def _load_unique_passages(limit: int | None) -> list[dict]:
    """Return deduplicated passage chunks from train + dev."""
    seen: dict[str, dict] = {}
    for split in ["train", "dev"]:
        path = DATA_DIR / f"{split}.json"
        with path.open() as f:
            examples = json.load(f)
        for ex in examples:
            fn = ex.get("filename")
            if not fn or fn in seen:
                continue
            ticker, year, page = _parse_filename(fn)
            text = _build_chunk_text(ex)
            if not text.strip():
                continue
            seen[fn] = {
                "chunk_id": fn,
                "ticker": ticker,
                "source_type": "10-K",
                "section_name": f"{ticker} {year} page_{page}",
                "fiscal_period": year,
                "filing_date": f"{year}-12-31",
                "text": text,
                "split": split,
            }
    chunks = list(seen.values())
    logger.info(f"Loaded {len(chunks)} unique passages (train+dev deduped by filename)")
    if limit:
        chunks = chunks[:limit]
        logger.info(f"Limited to {len(chunks)} for smoke test")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="Cap corpus size (smoke test)")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help="Output directory for FAISS + BM25 artifacts",
    )
    args = parser.parse_args()

    chunks = _load_unique_passages(args.limit)

    logger.info(f"Embedding {len(chunks)} passages with Titan V2…")
    embedder = TitanEmbedder()
    vectors = embedder.embed_batch([c["text"] for c in chunks], show_progress=True)

    logger.info(f"Building FAISS index at {args.index_dir}")
    args.index_dir.mkdir(parents=True, exist_ok=True)
    faiss_mgr = FAISSManager(index_dir=args.index_dir)
    faiss_mgr.build_index(vectors, chunks)
    faiss_mgr.save()

    logger.info(f"Building BM25 index at {args.index_dir}")
    bm25 = BM25Index(index_dir=args.index_dir)
    bm25.build(chunks)
    bm25.save()

    logger.info(f"Done. {len(chunks)} passages indexed at {args.index_dir}")


if __name__ == "__main__":
    main()
