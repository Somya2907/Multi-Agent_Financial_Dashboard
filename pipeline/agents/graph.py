"""LangGraph state graph for the multi-agent financial RAG pipeline.

Graph topology
--------------

  START
    │
  planner ─── metrics_only_mode=True ───► financial_metrics ──► risk_synthesis
    │                                                                   ▲
    │ (default)                                                          │
    ▼                                                                    │
  retrieval ──► fan-out ──► [financial_metrics, qualitative_analysis] ──┤
                                                                         │
                       (both branches run in parallel,                   │
                        converge at risk_synthesis)                      │
                                                                         │
                                                                  risk_synthesis
                                                                         │
                                                                 synthesize_answer
                                                                         │
                                                                      critic
                                                                         │
                                                                        END
"""

import json
import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from pipeline.agents.state import GraphState
from pipeline.agents.planner import run_planner
from pipeline.agents.retrieval import run_retrieval
from pipeline.agents.financial_metrics import run_financial_metrics
from pipeline.agents.qualitative import run_qualitative_analysis
from pipeline.agents.risk_synthesis import run_risk_synthesis
from pipeline.agents.critic import run_critic
from pipeline.reasoning.answer_generator import generate_answer

logger = logging.getLogger(__name__)


# ── Answer synthesis node ────────────────────────────────────────────────────

def _synthesize_answer(state: dict) -> dict:
    """Merge all agent outputs and generate the final grounded answer."""
    query = state["query"]
    context = state.get("context", "")
    all_metrics = state.get("metrics", {})
    tickers = state.get("tickers", [])

    # ── Inject metrics into context ──────────────────────────────────────────
    if all_metrics:
        is_comparison = len(all_metrics) > 1

        metrics_lines = ["\n=== FINANCIAL METRICS ==="]

        if is_comparison:
            # Rank by current_ratio ascending: lower ratio = higher liquidity risk
            def _cr(item):
                v = item[1].get("current_ratio")
                return v if isinstance(v, (int, float)) else float("inf")

            ranked = sorted(all_metrics.items(), key=_cr)
            metrics_lines.append(
                "Ranked by Current Ratio (ascending = higher liquidity risk):"
            )
            for rank, (ticker, m) in enumerate(ranked, 1):
                cr = m.get("current_ratio")
                de = m.get("debt_to_equity")
                cr_str = f"{cr:.3f}" if isinstance(cr, float) else ("N/A" if cr is None else str(cr))
                de_str = f"{de:.3f}" if isinstance(de, float) else ("N/A" if de is None else str(de))
                metrics_lines.append(
                    f"  {rank}. {ticker}: Current Ratio = {cr_str}, Debt-to-Equity = {de_str}"
                )
            # Also include all other metrics per ticker
            for ticker, m in all_metrics.items():
                extras = {
                    k: v for k, v in m.items()
                    if k not in {"current_ratio", "debt_to_equity"}
                    and not isinstance(v, dict)
                }
                if extras:
                    metrics_lines.append(f"\n  {ticker} additional metrics:")
                    for k, v in extras.items():
                        val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                        metrics_lines.append(f"    {k}: {val_str}")
        else:
            # Single company — flat display
            for ticker, m in all_metrics.items():
                metrics_lines.append(f"{ticker}:")
                for k, v in m.items():
                    if isinstance(v, float):
                        metrics_lines.append(f"  {k}: {v:.3f}")
                    elif isinstance(v, list) and not v:
                        continue
                    else:
                        metrics_lines.append(f"  {k}: {v}")

        metrics_text = "\n".join(metrics_lines)
        context += metrics_text
        logger.info(
            f"[Synthesize] Injected metrics for {list(all_metrics.keys())} into context"
        )

    # ── Fallback context when retrieval found nothing ────────────────────────
    if not context.strip() and not all_metrics:
        logger.warning("[Synthesize] No context and no metrics — answer may be empty")

    if not state.get("retrieved_chunks") and all_metrics:
        logger.info("[Synthesize] Metrics-only mode: no retrieved passages")

    # ── Qualitative analysis block ───────────────────────────────────────────
    qual = state.get("qualitative_analysis")
    if qual and "error" not in qual:
        context += (
            "\n\n=== QUALITATIVE ANALYSIS ===\n"
            + json.dumps(qual, indent=2)
        )

    # ── Risk synthesis block ─────────────────────────────────────────────────
    risk = state.get("risk_synthesis")
    if risk and "error" not in risk:
        risk_lines = ["\n\n=== RISK SYNTHESIS ==="]
        overall_level = risk.get("overall_risk_level", "UNKNOWN")
        overall_score = risk.get("overall_risk_score", 0.0)
        risk_lines.append(
            f"Overall Risk Level: {overall_level} (score: {overall_score:.2f})"
        )
        risk_lines.append(f"Summary: {risk.get('summary', '')}")
        risk_lines.append("\nRisk Category Breakdown:")
        for cat in (
            "market_risk", "credit_risk", "liquidity_risk",
            "regulatory_risk", "macroeconomic_risk",
        ):
            cat_data = risk.get(cat, {})
            if cat_data:
                sev = cat_data.get("severity", "?")
                score = cat_data.get("score", 0.0)
                evidence = "; ".join(cat_data.get("evidence", []))
                risk_lines.append(
                    f"  {cat.replace('_', ' ').title()}: {sev} "
                    f"({score:.2f}) — {evidence}"
                )
        context += "\n".join(risk_lines)
        logger.info(
            f"[Synthesize] Injected risk synthesis (overall={overall_level}) into context"
        )

    # ── Pass metrics to LLM prompt (separate from context) ──────────────────
    # For single company pass flat dict; for multi-company pass as-is (all dicts).
    if len(tickers) == 1 and tickers[0] in all_metrics:
        metrics_for_llm = all_metrics[tickers[0]]
    elif all_metrics:
        metrics_for_llm = all_metrics   # {ticker: {...}, ...} — handled by _format_metrics_block
    else:
        metrics_for_llm = None

    logger.info("[Synthesize] Generating final answer")
    answer = generate_answer(query, context, metrics_for_llm)
    return {"final_answer": answer}


# ── Conditional routing functions ────────────────────────────────────────────

def _route_after_planner(
    state: dict,
) -> Literal["financial_metrics", "retrieval"]:
    """Skip retrieval entirely for pure-metrics queries."""
    if state.get("metrics_only_mode"):
        logger.info(
            "[Route] metrics_only_mode=True — routing planner → financial_metrics"
        )
        return "financial_metrics"
    return "retrieval"


def _route_after_retrieval(state: dict) -> list[str]:
    """Fan out to metrics and/or qualitative in parallel, or skip both.

    Returns a list of next nodes so LangGraph executes them concurrently.
    Both branches converge at risk_synthesis via a direct edge.
    """
    chunks = state.get("retrieved_chunks", [])
    single_company = len(state.get("tickers", [])) == 1
    needs_metrics = state.get("requires_metrics", False) or single_company
    needs_qual = (state.get("requires_qualitative", False) or single_company) and chunks

    targets: list[str] = []
    if needs_metrics:
        targets.append("financial_metrics")
        if not chunks:
            logger.info(
                "[Route] 0 chunks retrieved + needs_metrics — metrics-only path"
            )
    if needs_qual:
        targets.append("qualitative_analysis")

    if not targets:
        return ["risk_synthesis"]

    logger.info(f"[Route] Fan-out after retrieval → {targets}")
    return targets


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the multi-agent financial RAG graph."""
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("planner", run_planner)
    workflow.add_node("retrieval", run_retrieval)
    workflow.add_node("financial_metrics", run_financial_metrics)
    workflow.add_node("qualitative_analysis", run_qualitative_analysis)
    workflow.add_node("risk_synthesis", run_risk_synthesis)
    workflow.add_node("synthesize_answer", _synthesize_answer)
    workflow.add_node("critic", run_critic)

    workflow.set_entry_point("planner")

    # planner → retrieval | financial_metrics (metrics_only shortcut)
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "retrieval": "retrieval",
            "financial_metrics": "financial_metrics",
        },
    )

    # retrieval → [financial_metrics, qualitative_analysis] (parallel fan-out)
    # or → risk_synthesis when neither agent is needed
    workflow.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "financial_metrics": "financial_metrics",
            "qualitative_analysis": "qualitative_analysis",
            "risk_synthesis": "risk_synthesis",
        },
    )

    # Both parallel branches converge at risk_synthesis
    workflow.add_edge("financial_metrics", "risk_synthesis")
    workflow.add_edge("qualitative_analysis", "risk_synthesis")

    # risk_synthesis → synthesize_answer (always)
    workflow.add_edge("risk_synthesis", "synthesize_answer")

    # synthesize_answer → critic → END
    workflow.add_edge("synthesize_answer", "critic")
    workflow.add_edge("critic", END)

    return workflow.compile()


# ── Singleton graph + public API ──────────────────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(
    query: str,
    ticker: str | None = None,
    year_filter: tuple[int, int] | None = None,
) -> dict:
    """Run the full multi-agent pipeline for a natural language financial query.

    Args:
        query:       Natural language question.
        ticker:      Optional ticker hint; bypasses planner's entity extraction
                     and forces single-company retrieval for this ticker.
        year_filter: Optional (min_year, max_year) to limit retrieved chunks
                     by fiscal_period year. Used by /analyze_company to
                     restrict to the last 2 years.

    Returns:
        Final GraphState dict with keys:
            final_answer, citations, metrics, qualitative_analysis,
            retrieved_chunks, query_type, tickers, metrics_only_mode.
    """
    initial_state: GraphState = {
        "query": query,
        "year_filter": year_filter,
        "query_type": "general",
        "tickers": [ticker] if ticker else [],
        "requires_metrics": False,
        "requires_qualitative": False,
        "metrics_only_mode": False,
        "retrieved_chunks": [],
        "context": "",
        "citations": [],
        "metrics": {},
        "qualitative_analysis": None,
        "risk_synthesis": None,
        "critique": None,
        "final_answer": "",
    }

    graph = _get_graph()
    result = graph.invoke(initial_state)
    return result
