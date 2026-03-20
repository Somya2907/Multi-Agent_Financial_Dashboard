"""Test the full post-RAG reasoning layer end-to-end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from pipeline.reasoning.query_engine import QueryEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def run(query: str, ticker: str | None = None, **kwargs):
    engine = QueryEngine(retrieval_k=20, min_score=0.45, max_context_chunks=12)
    result = engine.answer(query, ticker=ticker, **kwargs)

    print(f"\n{'='*80}")
    print(f"QUERY : {result.query}")
    if ticker:
        print(f"FILTER: ticker={ticker}")
    print(f"CHUNKS: {result.retrieved_chunks} retrieved → {result.chunks_after_filter} after filter")
    print(f"{'='*80}")
    print(result.answer)

    if result.metrics:
        print("\n--- Financial Metrics ---")
        for k, v in result.metrics.items():
            if k in ("ticker", "periods_covered"):
                print(f"  {k}: {v}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.2f}")

    if result.citations:
        print("\n--- Sources ---")
        for c in result.citations:
            print(
                f"  [{c['ticker']} | {c['source_type']} | {c['section_name']} | "
                f"score={c['score']:.3f}]"
            )


if __name__ == "__main__":
    # Single-company: factual
    run("What are Apple's main risk factors related to supply chain?", ticker="AAPL")

    # Single-company: metrics + context
    run("How has Microsoft's revenue growth trended and what is the debt profile?", ticker="MSFT")

    # Transcript-based
    run("What did NVIDIA's management say about AI demand and margins?", ticker="NVDA")

    # Cross-company (no ticker filter)
    run("Which companies have the highest liquidity risk based on their current ratio?")
