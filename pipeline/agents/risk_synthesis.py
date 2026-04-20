"""Risk Synthesis Agent: derives structured risk profiles from already-computed agent outputs.

Hybrid approach:
  • XBRL metrics (current_ratio → liquidity, debt_to_equity → credit)
    produce deterministic severity + analyst-prose evidence.
  • Keyword scoring over qualitative themes + retrieved chunks assigns
    severity for market / regulatory / macroeconomic risks.
  • A single LLM call then rewrites the matched passages for those three
    categories into concise, trader-friendly bullet points explaining what
    specifically drives the assigned severity.

Output key: risk_synthesis
"""

import json
import logging
import re

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Keyword sets per risk category ───────────────────────────────────────────

_MARKET_TERMS = {
    "competition", "competitive", "market share", "pricing pressure", "demand",
    "revenue", "customer", "product", "technology", "disruption", "innovation",
}
_REGULATORY_TERMS = {
    "regulation", "regulatory", "compliance", "legal", "litigation", "sec",
    "lawsuit", "enforcement", "tax", "legislation", "government", "antitrust",
    "privacy", "data protection",
}
_MACRO_TERMS = {
    "interest rate", "inflation", "supply chain", "recession", "gdp",
    "commodity", "currency", "foreign exchange", "fx", "macro", "economy",
    "geopolitical", "tariff", "trade", "manufacturing",
}


def _score_to_severity(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


# ── Liquidity risk (current_ratio) ───────────────────────────────────────────

def _assess_liquidity(metrics: dict) -> dict:
    evidence, scores = [], []

    for ticker, m in metrics.items():
        cr = m.get("current_ratio")
        if cr is None:
            continue
        if cr < 1.0:
            scores.append(0.85)
            evidence.append(
                f"{ticker} current ratio {cr:.2f} is below 1.0 — short-term liabilities "
                "exceed current assets, indicating near-term liquidity stress"
            )
        elif cr < 1.5:
            scores.append(0.50)
            evidence.append(
                f"{ticker} current ratio {cr:.2f} provides a thin liquidity buffer; "
                "deterioration could impair short-term obligations"
            )
        else:
            scores.append(0.20)
            evidence.append(
                f"{ticker} current ratio {cr:.2f} indicates adequate coverage of "
                "short-term liabilities with current assets"
            )

    if not scores:
        return {
            "severity": "MEDIUM",
            "score": 0.50,
            "evidence": ["Insufficient balance sheet data to assess liquidity risk"],
        }

    avg = round(sum(scores) / len(scores), 2)
    return {"severity": _score_to_severity(avg), "score": avg, "evidence": evidence[:3]}


# ── Credit risk (debt_to_equity) ─────────────────────────────────────────────

def _assess_credit(metrics: dict) -> dict:
    evidence, scores = [], []

    for ticker, m in metrics.items():
        de = m.get("debt_to_equity")
        if de is None:
            continue
        if de > 3.0:
            scores.append(0.80)
            evidence.append(
                f"{ticker} debt-to-equity {de:.2f} reflects high financial leverage, "
                "increasing sensitivity to interest rate and refinancing risk"
            )
        elif de > 1.5:
            scores.append(0.50)
            evidence.append(
                f"{ticker} debt-to-equity {de:.2f} represents moderate leverage "
                "with manageable but notable credit exposure"
            )
        elif de >= 0:
            scores.append(0.25)
            evidence.append(
                f"{ticker} debt-to-equity {de:.2f} indicates a conservative capital structure "
                "with limited credit risk"
            )
        else:
            scores.append(0.35)
            evidence.append(
                f"{ticker} negative shareholders' equity warrants monitoring of solvency metrics"
            )

    if not scores:
        return {
            "severity": "MEDIUM",
            "score": 0.50,
            "evidence": ["Insufficient balance sheet data to assess credit risk"],
        }

    avg = round(sum(scores) / len(scores), 2)
    return {"severity": _score_to_severity(avg), "score": avg, "evidence": evidence[:3]}


# ── Keyword-driven risk from qualitative themes + chunk sections ──────────────

def _extract_snippet(text: str, keyword: str, width: int = 140) -> str | None:
    """Return a ~width-char window around the first occurrence of keyword."""
    if not text or not keyword:
        return None
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return None
    start = max(0, idx - width // 3)
    end = min(len(text), idx + len(keyword) + (2 * width) // 3)
    snippet = text[start:end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _keyword_score(
    themes: list[str],
    chunks: list[dict],
    keywords: set,
    section_hint: str = "",
    category_label: str = "this category",
) -> tuple[float, list[str], list[dict]]:
    """Score a risk category and return driver-style evidence strings.

    Evidence explains WHY the severity was assigned:
      - which specific keywords drove the match
      - which document sections they appeared in
      - a representative quote snippet
    """
    matched_themes = [t for t in themes if any(k in t.lower() for k in keywords)]

    # Scan chunks: record which keywords fired, where, and a snippet.
    matched_keywords: set[str] = set()
    matched_sections: set[str] = set()
    matched_chunks: list[dict] = []
    example_snippet: str | None = None
    example_source: str | None = None
    chunk_hits = 0

    for c in chunks[:20]:
        body = c.get("text", "") or ""
        section = c.get("section_name", "") or ""
        haystack = (section + " " + body[:400]).lower()

        fired_here = [k for k in keywords if k in haystack]
        section_hit = bool(section_hint) and section_hint in haystack

        if fired_here or section_hit:
            chunk_hits += 1
            matched_keywords.update(fired_here)
            matched_chunks.append(c)
            if section:
                matched_sections.add(section)
            if example_snippet is None and fired_here:
                snip = _extract_snippet(body, fired_here[0])
                if snip:
                    example_snippet = snip
                    example_source = (
                        f"{c.get('ticker', '')} {c.get('source_type', '')} "
                        f"{section}".strip()
                    )

    theme_score = min(len(matched_themes) * 0.20, 0.60)
    chunk_score = min(chunk_hits * 0.05, 0.30)
    score = round(min(theme_score + chunk_score, 0.85), 2)
    severity = _score_to_severity(score)

    evidence: list[str] = []

    # Driver line — why this severity was assigned.
    if matched_keywords or matched_themes:
        driver_terms = sorted(matched_keywords)[:4]
        if matched_themes:
            # Prefer qualitative themes over raw keywords when available
            driver_terms = list(dict.fromkeys(matched_themes[:2] + driver_terms))[:4]
        terms_str = ", ".join(f'"{t}"' for t in driver_terms)
        section_str = (
            f" in {', '.join(sorted(matched_sections)[:2])}"
            if matched_sections else ""
        )
        evidence.append(
            f"{severity} severity driven by recurring {category_label} language "
            f"({terms_str}){section_str} across {chunk_hits} retrieved passage"
            f"{'s' if chunk_hits != 1 else ''}."
        )
    else:
        evidence.append(
            f"{severity} severity — no material {category_label} language "
            "surfaced in the retrieved Risk Factors or MD&A sections."
        )

    # Quote line — concrete evidence supporting the driver.
    if example_snippet:
        src = f" [{example_source}]" if example_source else ""
        evidence.append(f'Example: "{example_snippet}"{src}')

    return score, evidence, matched_chunks


def _assess_market(qual: dict, chunks: list[dict]) -> dict:
    themes = qual.get("risk_themes", []) or []
    key_themes = qual.get("key_themes", []) or []
    all_themes = themes + key_themes

    score, evidence, matched_chunks = _keyword_score(
        all_themes, chunks, _MARKET_TERMS, "md&a",
        category_label="competitive / market-demand",
    )

    # Boost from negative sentiment
    sentiment = qual.get("sentiment_score") or 0
    if sentiment < -0.2:
        score = min(score + 0.15, 0.85)
        evidence.append(
            f"Severity boosted by negative management tone "
            f"(sentiment score {sentiment:.2f}) signalling demand or pricing headwinds."
        )
    elif sentiment > 0.3:
        score = max(score - 0.10, 0.10)

    score = round(score, 2)
    return {
        "severity": _score_to_severity(score), "score": score,
        "evidence": evidence[:3], "_matched_chunks": matched_chunks,
    }


def _assess_regulatory(qual: dict, chunks: list[dict]) -> dict:
    themes = qual.get("risk_themes", []) or []
    score, evidence, matched_chunks = _keyword_score(
        themes, chunks, _REGULATORY_TERMS, "legal",
        category_label="regulatory / legal / compliance",
    )
    return {
        "severity": _score_to_severity(score), "score": score,
        "evidence": evidence[:3], "_matched_chunks": matched_chunks,
    }


def _assess_macro(qual: dict, chunks: list[dict]) -> dict:
    themes = (qual.get("risk_themes", []) or []) + (qual.get("key_themes", []) or [])
    score, evidence, matched_chunks = _keyword_score(
        themes, chunks, _MACRO_TERMS, "supply chain",
        category_label="macroeconomic / supply-chain",
    )
    return {
        "severity": _score_to_severity(score), "score": score,
        "evidence": evidence[:3], "_matched_chunks": matched_chunks,
    }


# ── LLM rewrite: turn matched chunks into analyst-style bullets ──────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


_REWRITE_SYSTEM = """\
You are an equity research analyst writing concise, actionable risk bullets \
for a trading desk. For each risk category below you receive:
  - the company ticker(s)
  - the assigned severity (LOW / MEDIUM / HIGH)
  - several retrieved passages from that company's 10-K / 10-Q filings
  - any relevant qualitative themes already extracted

Produce 1–2 bullet points per category that explain, in trader-friendly \
prose, WHAT SPECIFICALLY drives the assigned severity. Rules:
  - Reference concrete facts from the passages (products, markets, \
    regulations, cost items, geographies, customers) — not generic phrases.
  - Prefer specific numbers or names from the excerpts when present.
  - Do NOT quote the passage verbatim at length; paraphrase into 1 sentence.
  - Do NOT mention keyword matching, section hits, or retrieval mechanics.
  - If the passages for a category are weak, say so plainly in one sentence \
    ("Limited evidence in the retrieved filings — severity reflects absence \
    of material disclosures.").
  - Each bullet ≤ 220 characters. No markdown, no leading dashes.

Return JSON only — no markdown fences:
{
  "market_risk":        ["...", "..."],
  "regulatory_risk":    ["...", "..."],
  "macroeconomic_risk": ["...", "..."]
}
"""


def _pack_category_context(cat_key: str, cat_data: dict, qual: dict) -> str:
    chunks = cat_data.get("_matched_chunks") or []
    lines = [
        f"Category: {cat_key}",
        f"Severity: {cat_data.get('severity')} (score {cat_data.get('score')})",
    ]
    themes = qual.get("risk_themes") or []
    if themes:
        lines.append(f"Qualitative risk themes: {', '.join(themes[:4])}")
    if not chunks:
        lines.append("Passages: none matched")
    else:
        lines.append("Passages:")
        for c in chunks[:4]:
            label = f"{c.get('ticker', '')} {c.get('source_type', '')} {c.get('section_name', '')}".strip()
            snippet = re.sub(r"\s+", " ", (c.get("text") or "")[:500]).strip()
            lines.append(f"  [{label}] {snippet}")
    return "\n".join(lines)


def _llm_rewrite_evidence(
    categories: dict,
    qual: dict,
    tickers: list[str],
) -> None:
    """Replace keyword-based evidence on market/regulatory/macro risks with \
    analyst-style bullets synthesised by an LLM from the matched passages.

    Mutates `categories` in place. On any failure, keeps the heuristic evidence.
    """
    target_keys = ("market_risk", "regulatory_risk", "macroeconomic_risk")

    # Skip entirely if none of the target categories matched any passages.
    if not any(categories[k].get("_matched_chunks") for k in target_keys):
        for k in target_keys:
            categories[k].pop("_matched_chunks", None)
        return

    sections = [f"Company: {', '.join(tickers) if tickers else 'unknown'}"]
    for k in target_keys:
        sections.append("")
        sections.append(_pack_category_context(k, categories[k], qual))
    user_message = "\n".join(sections)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=6000,
        )
        content = (response.choices[0].message.content or "").strip()
        finish = response.choices[0].finish_reason
        reasoning = getattr(
            response.usage.completion_tokens_details, "reasoning_tokens", 0
        )
        logger.info(
            f"[RiskSynthesis] Rewrite call — finish={finish}, "
            f"reasoning_tokens={reasoning}, content_len={len(content)}"
        )
        if not content:
            raise ValueError(
                f"empty LLM output (finish={finish}, reasoning_tokens={reasoning})"
            )

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # If the model wrapped JSON in prose, extract the outermost {...} block
        if not content.startswith("{"):
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                content = m.group(0)

        parsed = json.loads(content)
    except Exception as e:
        logger.warning(
            f"[RiskSynthesis] Evidence rewrite failed ({e}); keeping heuristic evidence"
        )
        for k in target_keys:
            categories[k].pop("_matched_chunks", None)
        return

    for k in target_keys:
        bullets = parsed.get(k) or []
        bullets = [b.strip() for b in bullets if isinstance(b, str) and b.strip()]
        if bullets:
            categories[k]["evidence"] = bullets[:3]
        categories[k].pop("_matched_chunks", None)

    logger.info("[RiskSynthesis] LLM evidence rewrite applied to market/regulatory/macro")


# ── Summary template ──────────────────────────────────────────────────────────

def _build_summary(
    tickers: list[str],
    categories: dict,
    qual: dict,
    metrics: dict,
    overall_level: str,
    overall_score: float,
) -> str:
    name = ", ".join(tickers) if tickers else "The company"
    tone = (qual.get("overall_tone") or "neutral").capitalize()

    # Find top two risk categories by score
    ordered = sorted(
        [
            ("liquidity", categories["liquidity_risk"]),
            ("credit", categories["credit_risk"]),
            ("market", categories["market_risk"]),
            ("regulatory", categories["regulatory_risk"]),
            ("macroeconomic", categories["macroeconomic_risk"]),
        ],
        key=lambda x: x[1]["score"],
        reverse=True,
    )
    top1_name, top1 = ordered[0]
    top2_name, top2 = ordered[1]

    sentences = [
        f"{name} presents an overall {overall_level} risk profile "
        f"(composite score {overall_score:.2f}/1.00), with {top1_name} risk "
        f"({top1['severity']}) and {top2_name} risk ({top2['severity']}) "
        f"as the most material near-term concerns."
    ]

    # Metrics signal sentence — pull current_ratio, debt_to_equity, net_margin
    m_parts: list[str] = []
    first_ticker = tickers[0] if tickers else None
    m = metrics.get(first_ticker) if first_ticker else None
    if isinstance(m, dict):
        cr = m.get("current_ratio")
        de = m.get("debt_to_equity")
        nm = m.get("net_margin")
        fcf_m = m.get("fcf_margin")
        if cr is not None:
            m_parts.append(f"current ratio {cr:.2f}")
        if de is not None:
            m_parts.append(f"debt-to-equity {de:.2f}")
        if nm is not None:
            m_parts.append(f"net margin {nm * 100:.1f}%")
        if fcf_m is not None and len(m_parts) < 3:
            m_parts.append(f"FCF margin {fcf_m * 100:.1f}%")
    if m_parts:
        sentences.append(
            "Key financial signals include " + ", ".join(m_parts[:3]) + "."
        )

    # Management tone + sentiment sentence
    tone_sent = f"Management tone reads as {tone.lower()}"
    if qual.get("sentiment_score") is not None:
        tone_sent += f" (sentiment score {qual.get('sentiment_score', 0):.2f})"
    themes = qual.get("risk_themes") or qual.get("key_themes") or []
    themes = [t for t in themes if isinstance(t, str) and t.strip()]
    if themes:
        tone_sent += f", with recurring emphasis on {', '.join(themes[:2])}"
    sentences.append(tone_sent + ".")

    # Top risk evidence sentence
    top_ev = top1["evidence"][0] if top1["evidence"] else ""
    if top_ev:
        ev_clean = top_ev.rstrip(".")
        sentences.append(f"On the {top1_name} side, {ev_clean[0].lower() + ev_clean[1:]}.")

    # Forward-looking cue if available — ensures ≥4 sentences when evidence is thin
    forward = qual.get("forward_looking") or []
    forward = [f for f in forward if isinstance(f, str) and f.strip()]
    if forward and len(sentences) < 5:
        fwd_clean = forward[0].rstrip(".")
        sentences.append(f"Looking ahead, {fwd_clean[0].lower() + fwd_clean[1:]}.")

    return " ".join(sentences)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_risk_synthesis(state: dict) -> dict:
    """Derive structured risk profile from metrics, qualitative analysis, and retrieved chunks.

    Pure rule-based — no LLM call. Instant execution.

    Returns:
        {"risk_synthesis": dict}
    """
    metrics = state.get("metrics") or {}
    qual = state.get("qualitative_analysis") or {}
    chunks = state.get("retrieved_chunks") or []
    tickers = state.get("tickers") or []

    has_data = bool(metrics or qual or chunks)
    if not has_data:
        logger.info("[RiskSynthesis] No input data — returning insufficient_data marker")
        return {"risk_synthesis": {"error": "insufficient_data"}}

    try:
        liquidity = _assess_liquidity(metrics)
        credit = _assess_credit(metrics)
        market = _assess_market(qual, chunks)
        regulatory = _assess_regulatory(qual, chunks)
        macroeconomic = _assess_macro(qual, chunks)

        categories = {
            "liquidity_risk": liquidity,
            "credit_risk": credit,
            "market_risk": market,
            "regulatory_risk": regulatory,
            "macroeconomic_risk": macroeconomic,
        }

        # Rewrite keyword-driven categories into analyst-style prose bullets.
        _llm_rewrite_evidence(categories, qual, tickers)

        overall_score = round(
            (liquidity["score"] + credit["score"] + market["score"]
             + regulatory["score"] + macroeconomic["score"]) / 5,
            2,
        )
        overall_level = _score_to_severity(overall_score)
        summary = _build_summary(tickers, categories, qual, metrics, overall_level, overall_score)

        result = {
            **categories,
            "overall_risk_level": overall_level,
            "overall_risk_score": overall_score,
            "summary": summary,
        }

        logger.info(
            f"[RiskSynthesis] Done (rule-based) — overall={overall_level} score={overall_score}"
        )
        return {"risk_synthesis": result}

    except Exception as e:
        logger.error(f"[RiskSynthesis] Failed: {e}")
        return {"risk_synthesis": {"error": str(e)}}
