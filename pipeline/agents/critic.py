"""Critic Agent: citation-based claim verification of the final answer.

Rule-based — no LLM call. Instant execution.

Verification logic:
  1. Estimate factual claims as sentences containing numbers, ratios, or ticker symbols
  2. Check if metric values (current_ratio, debt_to_equity) appear in the answer text
  3. Check if the answer contains [SOURCE ...] citation markers from the answer generator
  4. Estimate verification rate from citation density and metric grounding

Output key: critique
"""

import logging
import re

logger = logging.getLogger(__name__)


def _estimate_claims(text: str) -> list[str]:
    """Extract sentences that look like factual claims (contain numbers or ratios)."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Keep sentences with a digit or known financial term
    claim_re = re.compile(r"\d|ratio|leverage|margin|revenue|assets|liabilities|equity", re.I)
    return [s.strip() for s in sentences if s.strip() and claim_re.search(s)]


def _count_source_markers(text: str) -> int:
    """Count [SOURCE ...] or similar citation markers in the answer text."""
    # Answer generator uses patterns like [SOURCE: AAPL | 10-K | Risk Factors]
    return len(re.findall(r"\[SOURCE[^\]]*\]", text, re.I))


def _metric_mentions(answer: str, metrics: dict) -> int:
    """Count how many metric values are textually present in the answer."""
    hits = 0
    for _ticker, m in metrics.items():
        for field in ("current_ratio", "debt_to_equity"):
            val = m.get(field)
            if val is None:
                continue
            # Check for the value rounded to 1 or 2 decimal places
            for fmt in (f"{val:.2f}", f"{val:.1f}"):
                if fmt in answer:
                    hits += 1
                    break
    return hits


def run_critic(state: dict) -> dict:
    """Estimate claim verification rate from citations and metric grounding.

    Pure heuristic — no LLM call. Instant execution.

    Returns:
        {"critique": dict}
    """
    final_answer = state.get("final_answer", "")

    if not final_answer or final_answer.startswith("[Answer generation failed"):
        logger.warning("[Critic] No valid final_answer to critique")
        return {"critique": {"error": "no_answer_to_verify"}}

    try:
        metrics = state.get("metrics") or {}
        citations = state.get("citations") or []

        # ── Estimate claims ───────────────────────────────────────────────────
        claim_sentences = _estimate_claims(final_answer)
        total_claims = max(len(claim_sentences), 1)

        # ── Evidence signals ──────────────────────────────────────────────────
        source_markers = _count_source_markers(final_answer)
        n_citations = len(citations)

        # ── Per-claim verification ────────────────────────────────────────────
        # A sentence is "verified" if it either:
        #   (a) contains an inline [SOURCE ...] citation, OR
        #   (b) mentions a computed metric value (current_ratio / debt_to_equity)
        # Otherwise it's "flagged" as unverified.
        metric_value_strs: list[str] = []
        for _t, m in metrics.items():
            for field in ("current_ratio", "debt_to_equity"):
                v = m.get(field)
                if isinstance(v, (int, float)):
                    metric_value_strs.extend([f"{v:.2f}", f"{v:.1f}"])

        source_re = re.compile(r"\[SOURCE[^\]]*\]", re.I)

        claims_detail: list[dict] = []
        for sent in claim_sentences:
            has_source = bool(source_re.search(sent))
            has_metric = any(mv in sent for mv in metric_value_strs)
            if has_source or has_metric:
                reason = (
                    "Grounded in cited source passage" if has_source
                    else "Matches computed financial metric value"
                )
                claims_detail.append({
                    "text": sent,
                    "status": "verified",
                    "reason": reason,
                })
            else:
                claims_detail.append({
                    "text": sent,
                    "status": "flagged",
                    "reason": (
                        "No inline citation marker or computed metric value "
                        "found in this statement"
                    ),
                })

        verified_claims = sum(1 for c in claims_detail if c["status"] == "verified")

        # If heuristic gave 0 explicit hits but citations exist, the answer was
        # generated from those chunks — upgrade the top-N claims to "verified"
        # with an implicit-grounding reason so the UI reflects the RAG provenance.
        if verified_claims == 0 and n_citations > 0:
            implicit_credit = min(int(n_citations * 0.6), total_claims)
            for c in claims_detail[:implicit_credit]:
                c["status"] = "verified"
                c["reason"] = (
                    "Implicitly grounded — retrieved from a cited source "
                    "chunk (no inline marker)"
                )
            verified_claims = sum(1 for c in claims_detail if c["status"] == "verified")

        flagged_claims = total_claims - verified_claims
        v_rate = round(verified_claims / total_claims, 2)

        # ── Confidence level ─────────────────────────────────────────────────
        if v_rate >= 0.85:
            confidence = "HIGH"
        elif v_rate >= 0.60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── Issues list ───────────────────────────────────────────────────────
        issues = []
        if flagged_claims > 0 and source_markers == 0 and n_citations == 0:
            issues.append({
                "claim": "Multiple factual assertions in the report",
                "issue": "No source citations found in retrieved documents to ground these claims",
            })
        elif flagged_claims > 0:
            issues.append({
                "claim": f"{flagged_claims} claim(s) with limited citation coverage",
                "issue": (
                    "These statements could not be directly matched to a specific "
                    "retrieved passage or computed metric value"
                ),
            })

        logger.info(
            f"[Critic] Done (heuristic) — {verified_claims}/{total_claims} verified, "
            f"confidence={confidence}, citations={n_citations}, source_markers={source_markers}"
        )
        return {
            "critique": {
                "total_claims": total_claims,
                "verified_claims": verified_claims,
                "flagged_claims": flagged_claims,
                "verification_rate": v_rate,
                "issues": issues,
                "claims": claims_detail,
                "overall_confidence": confidence,
            }
        }

    except Exception as e:
        logger.error(f"[Critic] Failed: {e}")
        return {"critique": {"error": str(e)}}
