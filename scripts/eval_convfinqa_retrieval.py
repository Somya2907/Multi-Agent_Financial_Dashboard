"""ConvFinQA retrieval evaluation.

Runs each ConvFinQA dev question through our Multi-HyDE retriever (FAISS + BM25
+ BGE reranker) over the ConvFinQA-only corpus built by
`scripts/build_convfinqa_corpus.py`. Gold evidence is the source passage the
question was drawn from — identified by ConvFinQA's `filename` field.

Metrics:
  - Recall@5 / @10 / @20   — gold passage appears in top-k
  - MRR@10                 — mean reciprocal rank of gold passage
  - Per-question-type breakdown (Type I vs Type II-first-turn)

Usage:
  python scripts/eval_convfinqa_retrieval.py
  python scripts/eval_convfinqa_retrieval.py --limit 50   # smoke test
  python scripts/eval_convfinqa_retrieval.py --k 20 --index-dir data/indexes/convfinqa
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

from openai import OpenAI  # noqa: E402

from config.settings import settings  # noqa: E402
from pipeline.embedding.embedder import TitanEmbedder  # noqa: E402
from pipeline.indexing.bm25_index import BM25Index  # noqa: E402
from pipeline.indexing.faiss_manager import FAISSManager  # noqa: E402
from pipeline.reasoning.hyde import MultiHyDERetriever  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("eval_convfinqa")
logger.setLevel(logging.INFO)

DEV_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "convfinqa" / "data" / "dev.json"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes" / "convfinqa"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "convfinqa_results.csv"


def _extract_query(ex: dict) -> tuple[str, str]:
    """Return (question_text, question_type). Falls back to qa_0 for Type II."""
    qa = ex.get("qa")
    if qa and qa.get("question"):
        return qa["question"], "type_i"
    qa0 = ex.get("qa_0")
    if qa0 and qa0.get("question"):
        return qa0["question"], "type_ii"
    return "", "unknown"


def _rank_of_gold(chunks: list[dict], gold_id: str) -> int:
    """1-indexed rank of the gold chunk, or 0 if not retrieved."""
    for i, c in enumerate(chunks, start=1):
        if c.get("chunk_id") == gold_id:
            return i
    return 0


def _load_dev(path: Path, limit: int | None) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def _build_retriever(index_dir: Path, hyde_model: str) -> MultiHyDERetriever:
    embedder = TitanEmbedder()
    faiss_mgr = FAISSManager(index_dir=index_dir)
    faiss_mgr.load()
    bm25 = BM25Index(index_dir=index_dir)
    bm25.load()
    llm = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return MultiHyDERetriever(
        faiss_manager=faiss_mgr,
        bm25_index=bm25,
        embedder=embedder,
        llm_client=llm,
        llm_model=hyde_model,
    )


def run_eval(
    dev_path: Path,
    index_dir: Path,
    output_csv: Path,
    k: int,
    limit: int | None,
    hyde_model: str,
) -> None:
    examples = _load_dev(dev_path, limit)
    logger.info(f"Loaded {len(examples)} dev examples from {dev_path}")

    logger.info(f"Building retriever from index at {index_dir} (HyDE model: {hyde_model})…")
    retriever = _build_retriever(index_dir, hyde_model)
    logger.info("Retriever ready.")

    per_query_rows: list[dict] = []
    by_type: dict[str, list[dict]] = defaultdict(list)

    for i, ex in enumerate(examples, start=1):
        qid = ex["id"]
        gold_id = ex["filename"]
        query, qtype = _extract_query(ex)
        if not query:
            logger.warning(f"[{qid}] no question text — skipping")
            continue

        t0 = time.time()
        try:
            chunks = retriever.run(query=query, final_k=k)
        except Exception as e:
            logger.error(f"[{qid}] retrieval failed: {e}")
            chunks = []
        elapsed_ms = int((time.time() - t0) * 1000)

        rank = _rank_of_gold(chunks, gold_id)
        r5 = 1.0 if 0 < rank <= 5 else 0.0
        r10 = 1.0 if 0 < rank <= 10 else 0.0
        r20 = 1.0 if 0 < rank <= 20 else 0.0
        rr10 = (1.0 / rank) if 0 < rank <= 10 else 0.0

        row = {
            "id": qid,
            "type": qtype,
            "gold_filename": gold_id,
            "query": query[:120],
            "rank": rank,
            "recall@5": r5,
            "recall@10": r10,
            "recall@20": r20,
            "rr@10": rr10,
            "n_retrieved": len(chunks),
            "latency_ms": elapsed_ms,
        }
        per_query_rows.append(row)
        by_type[qtype].append(row)

        status = "✓" if r5 else ("·" if r10 else "✗")
        print(
            f"[{i:3d}/{len(examples)}] {status} rank={rank or '—':<3} "
            f"R@5={r5:.0f} R@10={r10:.0f} R@20={r20:.0f} "
            f"RR@10={rr10:.2f} ({elapsed_ms}ms) :: {query[:60]}"
        )

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _avg(rows: list[dict], field: str) -> float:
        return (sum(r[field] for r in rows) / len(rows)) if rows else 0.0

    print("\n" + "=" * 78)
    print(f"{'Type':<12} {'N':>4} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'MRR@10':>9}")
    print("-" * 78)
    for t in sorted(by_type):
        rows = by_type[t]
        print(
            f"{t:<12} {len(rows):>4d} "
            f"{_avg(rows, 'recall@5'):>8.2%} "
            f"{_avg(rows, 'recall@10'):>8.2%} "
            f"{_avg(rows, 'recall@20'):>8.2%} "
            f"{_avg(rows, 'rr@10'):>9.3f}"
        )
    print("-" * 78)
    print(
        f"{'OVERALL':<12} {len(per_query_rows):>4d} "
        f"{_avg(per_query_rows, 'recall@5'):>8.2%} "
        f"{_avg(per_query_rows, 'recall@10'):>8.2%} "
        f"{_avg(per_query_rows, 'recall@20'):>8.2%} "
        f"{_avg(per_query_rows, 'rr@10'):>9.3f}"
    )
    print("=" * 78)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)
    print(f"\nPer-query results written to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test cap")
    parser.add_argument(
        "--hyde-model",
        type=str,
        default="gpt-4o-mini",
        help="Model used for Multi-HyDE query/hypothesis generation (reranker is unaffected)",
    )
    args = parser.parse_args()

    run_eval(
        args.dev, args.index_dir, args.output_csv,
        k=args.k, limit=args.limit, hyde_model=args.hyde_model,
    )


if __name__ == "__main__":
    main()
