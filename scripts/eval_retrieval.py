"""Retrieval evaluation harness.

Runs each query in data/eval/retrieval_eval.jsonl through the Multi-HyDE
retriever and scores the top-k results against metadata-level relevance
judgments (ticker + source_type + section keywords).

Metrics reported:
  - Recall@5 / @10 / @20   — fraction of queries with ≥1 relevant chunk in top-k
  - MRR@10                 — mean reciprocal rank of first relevant chunk in top-10
  - Per-category breakdown

Writes per-query results to data/eval/retrieval_results.csv and prints a
summary table to stdout.

Usage:
  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --eval-set data/eval/retrieval_eval.jsonl
  python scripts/eval_retrieval.py --k 20 --limit 10   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.reasoning.hyde import MultiHyDERetriever  # noqa: E402
from pipeline.indexing.faiss_manager import FAISSManager  # noqa: E402
from pipeline.indexing.bm25_index import BM25Index  # noqa: E402
from pipeline.embedding.embedder import TitanEmbedder  # noqa: E402
from openai import OpenAI  # noqa: E402
from config.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval_retrieval")
logger.setLevel(logging.INFO)


# ── Relevance judgment ────────────────────────────────────────────────────────

def _is_relevant(chunk: dict, relevance: dict) -> bool:
    """A chunk is relevant if it matches every declared constraint:
      - ticker in relevance["tickers"]
      - source_type in relevance["source_types"]
      - any section keyword matches section_name OR first 300 chars of text
    """
    if chunk.get("ticker") not in set(relevance.get("tickers", [])):
        return False
    if chunk.get("source_type") not in set(relevance.get("source_types", [])):
        return False

    keywords = [k.lower() for k in relevance.get("section_keywords", [])]
    if not keywords:
        return True

    haystack = (
        (chunk.get("section_name", "") or "").lower()
        + " "
        + (chunk.get("text", "") or "")[:300].lower()
    )
    return any(k in haystack for k in keywords)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _recall_at_k(rels: list[bool], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def _rr_at_k(rels: list[bool], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_eval(path: Path) -> list[dict]:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            entries.append(json.loads(line))
    return entries


# ── Retriever setup ───────────────────────────────────────────────────────────

def _build_retriever() -> MultiHyDERetriever:
    embedder = TitanEmbedder()
    faiss = FAISSManager()
    faiss.load()
    bm25 = BM25Index()
    bm25.load()
    llm = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return MultiHyDERetriever(
        faiss_manager=faiss,
        bm25_index=bm25,
        embedder=embedder,
        llm_client=llm,
        llm_model=settings.openai_llm_model,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eval(
    eval_path: Path,
    output_csv: Path,
    k: int = 20,
    limit: int | None = None,
) -> None:
    entries = _load_eval(eval_path)
    if limit:
        entries = entries[:limit]

    logger.info(f"Loaded {len(entries)} eval queries from {eval_path}")
    logger.info("Building retriever (loading FAISS + BM25 indexes)…")
    retriever = _build_retriever()
    logger.info("Retriever ready.")

    per_query_rows: list[dict] = []
    by_category: dict[str, list[dict]] = defaultdict(list)

    for i, entry in enumerate(entries, start=1):
        qid = entry["query_id"]
        query = entry["query"]
        category = entry.get("category", "uncategorized")
        ticker_filter = entry.get("ticker_filter")
        relevance = entry["relevance"]

        t0 = time.time()
        try:
            chunks = retriever.run(
                query=query,
                ticker_filter=ticker_filter,
                final_k=k,
            )
        except Exception as e:
            logger.error(f"[{qid}] retrieval failed: {e}")
            chunks = []
        elapsed_ms = int((time.time() - t0) * 1000)

        rels = [_is_relevant(c, relevance) for c in chunks]
        r5 = _recall_at_k(rels, 5)
        r10 = _recall_at_k(rels, 10)
        r20 = _recall_at_k(rels, 20)
        mrr10 = _rr_at_k(rels, 10)
        first_rank = next((idx + 1 for idx, r in enumerate(rels) if r), 0)
        n_relevant = sum(rels)

        row = {
            "query_id": qid,
            "category": category,
            "ticker_filter": ticker_filter or "",
            "query": query,
            "n_retrieved": len(chunks),
            "n_relevant": n_relevant,
            "first_rel_rank": first_rank,
            "recall@5": r5,
            "recall@10": r10,
            "recall@20": r20,
            "rr@10": mrr10,
            "latency_ms": elapsed_ms,
        }
        per_query_rows.append(row)
        by_category[category].append(row)

        status = "✓" if r5 else ("·" if r10 else "✗")
        print(
            f"[{i:2d}/{len(entries)}] {status} {qid:<14} "
            f"R@5={r5:.0f} R@10={r10:.0f} R@20={r20:.0f} "
            f"RR@10={mrr10:.2f} rank1={first_rank or '—'} "
            f"nRel={n_relevant}/{len(chunks)} ({elapsed_ms}ms)"
        )

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _avg(rows: list[dict], field: str) -> float:
        if not rows:
            return 0.0
        return sum(r[field] for r in rows) / len(rows)

    print("\n" + "=" * 72)
    print(f"{'Category':<18} {'N':>4} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'MRR@10':>8}")
    print("-" * 72)
    for cat in sorted(by_category):
        rows = by_category[cat]
        print(
            f"{cat:<18} {len(rows):>4d} "
            f"{_avg(rows, 'recall@5'):>7.2%} "
            f"{_avg(rows, 'recall@10'):>7.2%} "
            f"{_avg(rows, 'recall@20'):>7.2%} "
            f"{_avg(rows, 'rr@10'):>8.3f}"
        )
    print("-" * 72)
    print(
        f"{'OVERALL':<18} {len(per_query_rows):>4d} "
        f"{_avg(per_query_rows, 'recall@5'):>7.2%} "
        f"{_avg(per_query_rows, 'recall@10'):>7.2%} "
        f"{_avg(per_query_rows, 'recall@20'):>7.2%} "
        f"{_avg(per_query_rows, 'rr@10'):>8.3f}"
    )
    print("=" * 72)

    # ── Write CSV ────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)
    print(f"\nPer-query results written to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval/retrieval_eval.jsonl"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/eval/retrieval_results.csv"),
    )
    parser.add_argument("--k", type=int, default=20, help="Top-k to retrieve (default: 20)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N queries (smoke test)")
    args = parser.parse_args()

    run_eval(args.eval_set, args.output_csv, k=args.k, limit=args.limit)


if __name__ == "__main__":
    main()
