"""Planner agent: classify query type and extract routing parameters.

Uses GPT-5 to determine:
- query_type: single_company | comparison | general
- tickers: which companies are mentioned
- requires_metrics: does the question need financial ratios / statements?
- requires_qualitative: does the question need sentiment / tone / themes?
"""

import json
import logging

from openai import OpenAI

from config.settings import settings
from config.companies import COMPANIES

logger = logging.getLogger(__name__)

_KNOWN_TICKERS = {c["ticker"] for c in COMPANIES}

# Queries that match these keywords + require_metrics → skip retrieval entirely
_METRICS_ONLY_KEYWORDS = {
    "liquidity", "current ratio", "debt-to-equity", "debt to equity",
    "leverage", "compare", "comparison", "rank", "ranking",
    "which company", "which companies", "highest risk", "lowest ratio",
}

# Build a company name → ticker lookup for the prompt
_COMPANY_LIST = ", ".join(
    f"{c['ticker']} ({c['name']})" for c in COMPANIES
)

_PLANNER_SYSTEM = f"""\
You are a financial query router. Analyze the user question and output JSON only — no markdown fences.

Classify the query into one of:
  "single_company"  — about one specific company
  "comparison"      — comparing two or more companies
  "general"         — broad market, sector, or no specific company

Known companies: {_COMPANY_LIST}

Determine:
  requires_metrics    — true if the question needs revenue, margins, ratios, debt, liquidity numbers
  requires_qualitative — true if the question needs management tone, sentiment, forward guidance, risk themes

Output (JSON only):
{{
  "query_type": "single_company" | "comparison" | "general",
  "tickers": ["AAPL"],
  "requires_metrics": true,
  "requires_qualitative": false
}}"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def run_planner(state: dict) -> dict:
    """Classify the query and extract tickers + routing flags.

    Returns a partial state update with:
        query_type, tickers, requires_metrics, requires_qualitative
    """
    query = state["query"]
    # If the caller already supplied tickers, honour them and skip planning
    if state.get("tickers"):
        logger.info(
            f"[Planner] Ticker hint provided: {state['tickers']} — "
            "running lightweight classification only"
        )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_completion_tokens=300,
        )
        content = (response.choices[0].message.content or "{}").strip()
        # Strip accidental markdown fences
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content)
    except Exception as e:
        logger.warning(f"[Planner] LLM call failed ({e}); using defaults")
        plan = {
            "query_type": "general",
            "tickers": [],
            "requires_metrics": False,
            "requires_qualitative": False,
        }

    # Validate tickers against known list; honour caller-supplied hint
    extracted = [t for t in plan.get("tickers", []) if t in _KNOWN_TICKERS]
    final_tickers = state.get("tickers") or extracted

    requires_metrics = bool(plan.get("requires_metrics", False))

    # Bypass retrieval for queries that are purely about financial ratios / comparisons
    q_lower = query.lower()
    metrics_only = requires_metrics and any(kw in q_lower for kw in _METRICS_ONLY_KEYWORDS)

    logger.info(
        f"[Planner] query_type={plan.get('query_type')} "
        f"tickers={final_tickers} "
        f"metrics={requires_metrics} "
        f"metrics_only={metrics_only} "
        f"qualitative={plan.get('requires_qualitative')}"
    )

    return {
        "query_type": plan.get("query_type", "general"),
        "tickers": final_tickers,
        "requires_metrics": requires_metrics,
        "requires_qualitative": bool(plan.get("requires_qualitative", False)),
        "metrics_only_mode": metrics_only,
    }
