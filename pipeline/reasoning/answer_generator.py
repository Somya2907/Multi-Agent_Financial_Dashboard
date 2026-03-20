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
You are a financial analyst AI assistant. You answer questions about companies
using ONLY the evidence provided in the context passages and financial metrics below.

Rules:
- Ground every claim in the provided context or metrics. Do not invent facts.
- When citing a passage, reference its [SOURCE] label.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and precise. Use numbers when available.
- Structure your answer with: Summary, Key Evidence, and (if applicable) Caveats.
"""


def _format_metrics_block(metrics: dict | None) -> str:
    if not metrics or "error" in metrics:
        return ""
    lines = ["=== STRUCTURED FINANCIAL METRICS ==="]
    for k, v in metrics.items():
        if k in ("ticker", "periods_covered"):
            lines.append(f"{k}: {v}")
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.2f}")
        else:
            lines.append(f"{k}: {v}")
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
