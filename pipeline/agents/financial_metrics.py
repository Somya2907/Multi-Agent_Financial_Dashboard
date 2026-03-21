"""Financial Metrics agent — reads XBRL balance sheet data and computes ratios.

Data source
-----------
  data/financials/{ticker}_balance_sheet.json
  Written by _save_xbrl_balance_sheets() in orchestrator.py during pipeline run.

  Format::

      {
          "current_assets":       143_566_000_000.0,
          "current_liabilities":  134_690_000_000.0,
          "total_assets":         364_980_000_000.0,
          "total_liabilities":    308_030_000_000.0,
          "shareholder_equity":    56_950_000_000.0
      }

This agent does NOT touch FAISS, retrieved chunks, or raw text.

Output
------
  {
      "metrics": {
          "AAPL": {
              "ticker":        "AAPL",
              "current_ratio":  1.066,
              "debt_to_equity": 5.162,
              "raw": {
                  "current_assets":       143_566_000_000.0,
                  ...
              }
          },
          ...
      }
  }

Ticker resolution
-----------------
  1. Explicit tickers in state["tickers"]
  2. Tickers inferred from state["retrieved_chunks"] (for single-company context)
  3. All known tickers  — when query_type is "comparison"/"general" or
     state["requires_metrics"] is True
"""

import json
import logging
from pathlib import Path

from config.companies import COMPANIES

logger = logging.getLogger(__name__)

_ALL_TICKERS = [c["ticker"] for c in COMPANIES]
_BS_DIR = Path(__file__).resolve().parents[2] / "data" / "financials"

# Fields required to compute ratios
_BS_FIELDS = (
    "current_assets",
    "current_liabilities",
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
)


# ── XBRL disk load ────────────────────────────────────────────────────────────

def _load_xbrl(ticker: str) -> dict | None:
    """Load XBRL balance sheet values from disk.

    Returns the raw dict on success, None when the file is missing or corrupt.
    """
    path = _BS_DIR / f"{ticker}_balance_sheet.json"
    if not path.exists():
        logger.warning(
            f"[FinMetrics] No XBRL balance sheet file for {ticker} "
            f"({path}) — run the pipeline first"
        )
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"[FinMetrics] Failed to load {path}: {exc}")
        return None


# ── Ratio computation ─────────────────────────────────────────────────────────

def _compute_ratios(ticker: str, raw: dict) -> dict:
    """Compute current_ratio and debt_to_equity with safe division.

    Missing or zero denominators produce None (not an error).
    """
    result: dict = {"ticker": ticker, "raw": {k: raw.get(k) for k in _BS_FIELDS}}

    ca = raw.get("current_assets")
    cl = raw.get("current_liabilities")
    if ca is not None and cl:
        result["current_ratio"] = round(ca / cl, 3)
    else:
        logger.warning(
            f"[FinMetrics] {ticker}: cannot compute current_ratio "
            f"(current_assets={ca}, current_liabilities={cl})"
        )

    tl = raw.get("total_liabilities")
    eq = raw.get("shareholder_equity")
    if tl is not None and eq:
        result["debt_to_equity"] = round(tl / eq, 3)
    else:
        logger.warning(
            f"[FinMetrics] {ticker}: cannot compute debt_to_equity "
            f"(total_liabilities={tl}, shareholder_equity={eq})"
        )

    logger.info(
        f"[FinMetrics] {ticker} "
        f"current_ratio={result.get('current_ratio')} "
        f"debt_to_equity={result.get('debt_to_equity')}"
    )
    return result


# ── Main agent node ───────────────────────────────────────────────────────────

def run_financial_metrics(state: dict) -> dict:
    """Compute financial ratios from XBRL balance sheet data for each relevant ticker.

    Does not access FAISS, retrieved chunks, or any raw text.

    Returns:
        {"metrics": {ticker: {"ticker": str, "current_ratio": float,
                               "debt_to_equity": float, "raw": dict}, ...}}
    """
    query_type = state.get("query_type", "general")
    tickers: list[str] = list(state.get("tickers") or [])

    # ── Ticker resolution ─────────────────────────────────────────────────────
    if not tickers:
        chunks: list[dict] = state.get("retrieved_chunks", [])
        if chunks:
            found = {c.get("ticker") for c in chunks if c.get("ticker")}
            tickers = sorted(found)
            logger.info(f"[FinMetrics] Tickers from retrieved chunks: {tickers}")
        elif query_type in ("comparison", "general") or state.get("requires_metrics"):
            tickers = list(_ALL_TICKERS)
            logger.info(
                f"[FinMetrics] '{query_type}' query — computing metrics "
                f"for all companies: {tickers}"
            )

    if not tickers:
        logger.info("[FinMetrics] No tickers to compute — returning empty metrics")
        return {"metrics": {}}

    all_metrics: dict = {}

    for ticker in tickers:
        try:
            raw = _load_xbrl(ticker)
            if raw is None:
                continue                         # warning already logged in _load_xbrl

            metrics = _compute_ratios(ticker, raw)

            # Only include the ticker if at least one ratio was computed
            if "current_ratio" not in metrics and "debt_to_equity" not in metrics:
                logger.warning(
                    f"[FinMetrics] {ticker}: no ratios computed — skipping"
                )
                continue

            all_metrics[ticker] = metrics

        except Exception as exc:
            logger.warning(f"[FinMetrics] Unexpected error for {ticker}: {exc}")

    if all_metrics:
        logger.info(
            f"[FinMetrics] Metrics ready for: {list(all_metrics.keys())}"
        )
    else:
        logger.warning(
            "[FinMetrics] No metrics computed — "
            "verify pipeline has been run and XBRL files exist in data/financials/"
        )

    return {"metrics": all_metrics}
