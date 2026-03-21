"""Qualitative Analysis agent: extract tone, sentiment, and strategic themes.

Operates only on transcript, earnings release, and news chunks — SEC filings
(10-K, 10-Q) are excluded because they are legalistic rather than expressive.
"""

import json
import logging

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

_QUALITATIVE_SOURCE_TYPES = {"transcript", "earnings_release", "news", "8-k", "8-K"}

_QUALITATIVE_SYSTEM = """\
You are a qualitative financial analyst. Analyze the provided text passages and return JSON only — no markdown.

Extract:
  overall_tone          — "positive" | "neutral" | "negative" | "mixed"
  sentiment_score       — float, -1.0 (very negative) to 1.0 (very positive)
  key_themes            — list of up to 5 main topics discussed
  forward_looking       — list of up to 3 forward-looking or guidance statements (verbatim quotes)
  risk_themes           — list of up to 3 key risks mentioned
  management_confidence — "high" | "moderate" | "low" based on language certainty

Output format (JSON only):
{
  "overall_tone": "positive",
  "sentiment_score": 0.6,
  "key_themes": ["AI demand", "margin expansion"],
  "forward_looking": ["We expect revenue to grow..."],
  "risk_themes": ["supply chain constraints"],
  "management_confidence": "high"
}"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def run_qualitative_analysis(state: dict) -> dict:
    """Analyse tone, sentiment, themes, and forward guidance from soft-text sources.

    Only uses transcript / earnings release / news chunks — skips 10-K/10-Q filings
    which are regulatory documents rather than expressive management commentary.

    Returns a partial state update with:
        qualitative_analysis: dict | None
    """
    chunks = state.get("retrieved_chunks", [])
    qual_chunks = [
        c for c in chunks
        if c.get("source_type", "") in _QUALITATIVE_SOURCE_TYPES
    ]

    if not qual_chunks:
        logger.info("[Qualitative] No transcript/news chunks — skipping")
        return {"qualitative_analysis": None}

    # Use up to 6 highest-scoring qualitative chunks
    qual_chunks = sorted(qual_chunks, key=lambda c: c.get("score", 0), reverse=True)[:6]
    qualitative_context = "\n\n".join(c["text"] for c in qual_chunks)

    tickers = state.get("tickers", [])
    ticker_label = ", ".join(tickers) if tickers else "the company"
    query = state.get("query", "")

    user_message = (
        f"Query: {query}\n"
        f"Company/Companies: {ticker_label}\n\n"
        f"Text passages:\n{qualitative_context}"
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": _QUALITATIVE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=1000,
        )
        content = (response.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
        logger.info(
            f"[Qualitative] tone={analysis.get('overall_tone')} "
            f"score={analysis.get('sentiment_score')} "
            f"themes={analysis.get('key_themes')}"
        )
    except Exception as e:
        logger.warning(f"[Qualitative] Analysis failed: {e}")
        analysis = {"error": str(e)}

    return {"qualitative_analysis": analysis}
