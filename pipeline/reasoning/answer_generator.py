"""Generate a final structured answer using GPT-5 (OpenAI) over retrieved context."""

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
You are a sell-side equity research analyst writing a desk-ready note for a
trading team. Your output is read on a dashboard, so it must be scannable,
grounded, and free of hedging filler.

Ground every claim in the provided context passages or computed metrics — do
not invent numbers, dates, or quotes. When a specific passage supports a claim,
reference its [SOURCE] label inline. If the evidence is insufficient for a
section, write "Insufficient evidence in the available filings." rather than
speculating.

Write the report in GitHub-flavored Markdown using these exact section headers
in this order. Omit a section only if it would be empty.

## Investment Thesis
2–3 sentences stating the overall financial posture and the single most
important takeaway for a trader. Lead with the conclusion.

## Financial Position
- 3–5 bullets covering liquidity (current ratio), leverage (debt/equity),
  capital structure, and any notable balance-sheet line items.
- Every bullet cites at least one specific number.
- Flag trend direction (↑ / ↓ / flat) when two periods are available.

## Risk Assessment
- One bullet per material risk category that has evidence in the context
  (liquidity, credit, market, regulatory, macroeconomic). Skip categories
  with no supporting evidence.
- Each bullet: **Category (Severity)** — specific driver with numbers, then
  [SOURCE] reference.

## Management Commentary & Sentiment
2–4 sentences on management tone, forward guidance, and recurring themes,
drawn from the qualitative analysis and earnings-call / MD&A passages.
Attribute quotes when you use them.

## Bottom Line for Traders
One short paragraph (2–3 sentences) translating the analysis into an actionable
read: what to watch, what's priced in, what would change the view.

Style rules:
- Use concrete numbers (ratios to 2 dp, dollar amounts with units).
- Prefer active voice and present tense.
- No generic caveats ("this is not financial advice", etc.).
- Do NOT include a separate "Summary" / "Key Evidence" / "Caveats" trio —
  the section structure above replaces it.
"""


def _format_single_company_metrics(metrics: dict) -> list[str]:
    """Format one company's metrics dict into display lines (no header)."""
    lines: list[str] = []

    bs_fields = {
        "current_assets", "current_liabilities",
        "total_assets", "total_liabilities", "shareholder_equity",
    }

    # Prominent ratios first
    cr = metrics.get("current_ratio")
    de = metrics.get("debt_to_equity")
    if cr is not None:
        lines.append(f"  Current Ratio: {cr:.3f}")
    if de is not None:
        lines.append(f"  Debt-to-Equity: {de:.3f}")

    # Raw balance sheet line items — check both flat structure and nested "raw" key
    flat_bs = {k: v for k, v in metrics.items() if k in bs_fields}
    raw_bs = flat_bs or {k: v for k, v in metrics.get("raw", {}).items() if k in bs_fields}
    if raw_bs:
        lines.append("  Balance Sheet Line Items ($ thousands):")
        for k, v in raw_bs.items():
            try:
                lines.append(f"    {k}: {v:,.0f}")
            except (TypeError, ValueError):
                lines.append(f"    {k}: {v}")

    # Other metrics
    skip = bs_fields | {
        "source", "current_ratio", "debt_to_equity",
        "ticker", "filing_date", "periods_covered", "raw",
    }
    for k, v in metrics.items():
        if k in skip:
            continue
        if k == "periods_covered":
            lines.append(f"  {k}: {v}")
        elif isinstance(v, float):
            lines.append(f"  {k}: {v:.2f}")
        else:
            lines.append(f"  {k}: {v}")

    return lines


def _format_metrics_block(metrics: dict | None) -> str:
    if not metrics or "error" in metrics:
        return ""

    # ── Multi-company: values are dicts keyed by ticker ───────────────────────
    if all(isinstance(v, dict) for v in metrics.values()):
        lines = ["=== STRUCTURED FINANCIAL METRICS (MULTI-COMPANY) ==="]
        for ticker, m in metrics.items():
            if not m:
                lines.append(f"\n{ticker}: [No data available]")
                continue
            lines.append(f"\n{ticker}:")
            lines.extend(_format_single_company_metrics(m))
        return "\n".join(lines)

    # ── Single company ────────────────────────────────────────────────────────
    lines = ["=== STRUCTURED FINANCIAL METRICS ==="]
    lines.extend(_format_single_company_metrics(metrics))
    return "\n".join(lines)


def generate_answer(
    query: str,
    context: str,
    metrics: dict | None = None,
) -> str:
    """Call the LLM to produce a grounded answer from context and metrics.

    Uses max_completion_tokens (not max_tokens) because GPT-5 is a reasoning
    model that consumes internal reasoning tokens before producing visible output.
    temperature is omitted — reasoning models do not support it.

    Args:
        query: The user's original question.
        context: Aggregated passage context from retrieved chunks.
        metrics: Optional structured financial metrics dict.

    Returns:
        The generated answer string.
    """
    metrics_block = _format_metrics_block(metrics)

    user_message = f"QUESTION: {query}\n\n"
    if metrics_block:
        user_message += f"{metrics_block}\n\n"
    user_message += f"=== RETRIEVED CONTEXT ===\n{context}"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=8000,  # GPT-5 needs budget for reasoning + output
        )
        content = response.choices[0].message.content or ""
        reasoning = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)
        output = response.usage.completion_tokens - reasoning
        logger.info(f"LLM: {reasoning} reasoning tokens, {output} output tokens")
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"[Answer generation failed: {e}]"
