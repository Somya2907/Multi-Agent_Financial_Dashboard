"""Lightweight follow-up handler — answers grounded in a prior analysis.

Unlike the full pipeline, this does NOT run retrieval or the multi-agent graph.
It reuses the state produced by the first /analyze_company call and makes a
single LLM call with the already-retrieved context, so follow-ups return in
seconds rather than the ~60s required for a full analysis.
"""

import json
import logging

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


_SYSTEM_PROMPT = """\
You are a sell-side equity research analyst answering a trader's follow-up
question about a company analysis that has already been generated. Use ONLY
the prior analysis, metrics, and retrieved passages below — do not invent
facts or rely on outside knowledge.

Format the response as a short, desk-ready analyst note in GitHub-flavored
Markdown. Pick the sections that actually apply to the question (skip the
rest) from:

## Direct Answer
1–3 sentences giving the specific answer up front, with the key number or
conclusion in bold.

## Supporting Evidence
- 2–4 bullets with the specific metrics, quotes, or passages that back the
  answer. Every bullet cites at least one concrete number or a [SOURCE] label.

## Trader Takeaway
1–2 sentences translating the answer into what it means for positioning or
what to watch next.

Style rules:
- Lead with the conclusion, not the reasoning.
- Use concrete numbers (ratios to 2 dp, dollar amounts with units).
- If the prior analysis lacks the information, say "Not covered by the
  current analysis — would require re-running retrieval with a narrower
  query." and stop.
- Out-of-scope topics (live prices, macro data not in documents): state the
  scope limit in one sentence.
- Do NOT use the generic "Summary / Key Evidence / Caveats" structure.
"""


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return ""
    lines = ["=== FINANCIAL METRICS ==="]
    for ticker, m in metrics.items():
        lines.append(f"\n{ticker}:")
        cr = m.get("current_ratio")
        de = m.get("debt_to_equity")
        if cr is not None:
            lines.append(f"  Current Ratio: {cr}")
        if de is not None:
            lines.append(f"  Debt-to-Equity: {de}")
        raw = m.get("raw") or {}
        for k, v in raw.items():
            if v is None:
                continue
            try:
                lines.append(f"  {k}: {v:,.0f}")
            except (TypeError, ValueError):
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _format_risk(risk: dict | None) -> str:
    if not risk or "error" in risk:
        return ""
    lines = ["=== RISK SYNTHESIS ==="]
    lines.append(f"Overall: {risk.get('overall_risk_level')} ({risk.get('overall_risk_score')})")
    lines.append(f"Summary: {risk.get('summary', '')}")
    for cat in ("liquidity_risk", "credit_risk", "market_risk", "regulatory_risk", "macroeconomic_risk"):
        c = risk.get(cat) or {}
        if c:
            ev = "; ".join(c.get("evidence", [])[:2])
            lines.append(f"{cat}: {c.get('severity')} ({c.get('score')}) — {ev}")
    return "\n".join(lines)


def _format_qualitative(qual: dict | None) -> str:
    if not qual or "error" in qual:
        return ""
    return "=== QUALITATIVE ANALYSIS ===\n" + json.dumps(
        {k: v for k, v in qual.items() if k != "error"}, indent=2
    )


def _format_chunks(chunks: list[dict], max_chunks: int = 8) -> str:
    """Take top N chunks, keep 400-char slices with [SOURCE ...] labels."""
    if not chunks:
        return ""
    lines = ["=== EVIDENCE PASSAGES ==="]
    for c in chunks[:max_chunks]:
        label = f"{c.get('ticker', '')} | {c.get('source_type', '')} | {c.get('section_name', '')}"
        snippet = (c.get("text") or "")[:400].strip()
        lines.append(f"\n[SOURCE: {label}]\n{snippet}")
    return "\n".join(lines)


def run_followup(query: str, prior_state: dict) -> dict:
    """Answer a follow-up question grounded in a prior analysis.

    Args:
        query: The user's follow-up question.
        prior_state: The GraphState returned by /analyze_company or /query.

    Returns:
        {"answer": str, "citations": list[dict]} — citations are reused from
        prior_state so the UI can keep displaying the same sources.
    """
    if not query.strip():
        return {"answer": "Please ask a specific follow-up question.", "citations": []}

    metrics = prior_state.get("metrics") or {}
    risk = prior_state.get("risk_synthesis")
    qual = prior_state.get("qualitative_analysis")
    chunks = prior_state.get("retrieved_chunks") or []
    citations = prior_state.get("citations") or []
    prior_answer = prior_state.get("final_answer") or ""
    tickers = prior_state.get("tickers") or []

    # Build compact context from prior analysis
    parts = []
    if tickers:
        parts.append(f"COMPANY: {', '.join(tickers)}")
    if prior_answer:
        parts.append("=== PRIOR ANALYSIS ===\n" + prior_answer)
    for block in (_format_metrics(metrics), _format_risk(risk), _format_qualitative(qual), _format_chunks(chunks)):
        if block:
            parts.append(block)

    context = "\n\n".join(parts)

    user_message = f"FOLLOW-UP QUESTION: {query}\n\n{context}"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=6000,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            finish = response.choices[0].finish_reason
            logger.warning(
                f"[Followup] Empty answer — finish_reason={finish} "
                f"(likely reasoning-token budget exhausted)"
            )
            answer = (
                "_The model ran out of reasoning budget before producing a "
                "visible answer. Try rephrasing the question more narrowly._"
            )
        reasoning = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)
        output = response.usage.completion_tokens - reasoning
        logger.info(
            f"[Followup] {reasoning} reasoning + {output} output tokens — "
            f"answered from {len(chunks)} chunks"
        )
    except Exception as e:
        logger.error(f"[Followup] LLM call failed: {e}")
        return {"answer": f"[Follow-up failed: {e}]", "citations": citations}

    return {"answer": answer, "citations": citations}
