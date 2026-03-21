"""Shared state TypedDict for the multi-agent LangGraph pipeline."""

from typing import TypedDict


class GraphState(TypedDict):
    """State passed between agents in the financial RAG graph.

    Populated incrementally as each agent runs:
      planner      → query_type, tickers, requires_metrics, requires_qualitative
      retrieval    → retrieved_chunks, context, citations
      fin_metrics  → metrics
      qualitative  → qualitative_analysis
      synthesize   → final_answer
    """

    # Input
    query: str

    # Planner outputs
    query_type: str          # "single_company" | "comparison" | "general"
    tickers: list[str]       # Validated tickers extracted from query
    requires_metrics: bool   # True when financial ratios / statements needed
    requires_qualitative: bool  # True when tone / sentiment / themes needed
    metrics_only_mode: bool  # True → skip retrieval, run metrics agent directly

    # Retrieval outputs
    retrieved_chunks: list[dict]
    context: str             # Formatted multi-source context for the LLM
    citations: list[dict]    # Source references aligned with context

    # Financial Metrics agent output — keyed by ticker
    metrics: dict            # {ticker: {metric_name: value, ...}}

    # Qualitative Analysis agent output
    qualitative_analysis: dict | None

    # Final synthesized answer
    final_answer: str
