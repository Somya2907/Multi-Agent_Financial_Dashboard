"""Test the multi-agent LangGraph financial RAG pipeline end-to-end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import logging
from pipeline.agents.graph import run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def display(result: dict):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"QUERY       : {result['query']}")
    print(f"QUERY TYPE  : {result.get('query_type', '?')}")
    print(f"TICKERS     : {result.get('tickers', [])}")
    print(f"AGENTS RUN  : "
          f"planner ✓  "
          f"retrieval ✓ ({len(result.get('retrieved_chunks', []))} chunks)  "
          f"{'metrics ✓  ' if result.get('metrics') else ''}"
          f"{'qualitative ✓' if result.get('qualitative_analysis') else ''}")
    print(sep)
    print(result.get("final_answer", "[no answer]"))

    if result.get("metrics"):
        print("\n--- Financial Metrics ---")
        for ticker, m in result["metrics"].items():
            print(f"  {ticker}:")
            for k, v in m.items():
                if k in ("ticker", "periods_covered", "source"):
                    print(f"    {k}: {v}")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")

    if result.get("qualitative_analysis") and "error" not in result["qualitative_analysis"]:
        qual = result["qualitative_analysis"]
        print("\n--- Qualitative Analysis ---")
        print(f"  tone            : {qual.get('overall_tone')}")
        print(f"  sentiment_score : {qual.get('sentiment_score')}")
        print(f"  key_themes      : {qual.get('key_themes')}")
        print(f"  management conf : {qual.get('management_confidence')}")
        if qual.get("forward_looking"):
            print("  forward-looking :")
            for s in qual["forward_looking"]:
                print(f"    • {s}")
        if qual.get("risk_themes"):
            print("  risk themes     :")
            for r in qual["risk_themes"]:
                print(f"    • {r}")

    if result.get("citations"):
        print("\n--- Sources ---")
        for c in result["citations"][:5]:
            print(
                f"  [{c.get('ticker')} | {c.get('source_type')} | "
                f"{c.get('section_name')} | score={c.get('score', 0):.3f}]"
            )


if __name__ == "__main__":
    # ── Test 1: Single company — factual + metrics ────────────────────────────
    display(run_query(
        "What are Apple's main supply chain risk factors?",
        ticker="AAPL",
    ))

    # ── Test 2: Single company — revenue + debt profile ───────────────────────
    display(run_query(
        "How has Microsoft's revenue growth trended and what is the debt profile?",
        ticker="MSFT",
    ))

    # ── Test 3: Transcript — sentiment + qualitative ──────────────────────────
    display(run_query(
        "What did NVIDIA's management say about AI demand and future margins?",
        ticker="NVDA",
    ))

    # ── Test 4: Cross-company — no ticker filter ──────────────────────────────
    display(run_query(
        "Which companies have the highest liquidity risk based on current ratio?",
    ))
