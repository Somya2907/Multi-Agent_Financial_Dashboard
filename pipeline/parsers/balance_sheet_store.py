"""In-memory store for balance sheet metrics.

Lifetime
--------
* During a pipeline run, orchestrator.py calls set_metrics() for each ticker
  immediately after parsing + computing metrics from the 10-K HTML.
* During query-only runs (index already built, pipeline not re-run),
  initialize_all() is called once at metrics-agent startup: it scans the
  saved JSON files (written by the orchestrator) and populates the store so
  the agent never makes per-query file reads.

Single source of truth
-----------------------
  balance_sheet_store._data = {
      "AAPL": {
          "current_assets": 143_566_000.0,
          "current_liabilities": 134_690_000.0,
          "total_assets":        364_980_000.0,
          "total_liabilities":   308_030_000.0,
          "shareholder_equity":   56_950_000.0,
          "current_ratio":         1.066,
          "debt_to_equity":        5.162,
          "source":               "balance_sheet_extraction",
      },
      "MSFT": { ... },
      ...
  }
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "financials"

# ── In-memory store ───────────────────────────────────────────────────────────
_data: dict[str, dict] = {}
_initialized: bool = False  # True after initialize_all() has run


# ── Write path (called by orchestrator during pipeline run) ───────────────────

def set_metrics(ticker: str, metrics: dict) -> None:
    """Store balance sheet metrics for a ticker (called by orchestrator).

    Args:
        ticker:  Company ticker symbol.
        metrics: Dict returned by compute_financial_metrics(); contains both
                 raw balance sheet line items and computed ratios.
    """
    _data[ticker] = metrics
    logger.debug(
        f"[BSStore] Stored for {ticker}: "
        f"current_ratio={metrics.get('current_ratio')}, "
        f"debt_to_equity={metrics.get('debt_to_equity')}, "
        f"fields={list(metrics.keys())}"
    )


# ── Read path (called by financial_metrics agent) ─────────────────────────────

def get_metrics(ticker: str) -> dict | None:
    """Return stored balance sheet metrics for a ticker.

    If the store was not populated during this process (query-only run),
    falls back to a one-time per-ticker disk load.

    Returns None (with a clear log message) if data is unavailable.
    """
    if ticker in _data:
        return _data[ticker]

    # Attempt lazy single-ticker load from disk
    if _try_load_from_disk(ticker):
        return _data.get(ticker)

    logger.info(
        f"[BSStore] Balance sheet section not found in SEC parsing for {ticker}"
    )
    return None


def loaded_tickers() -> list[str]:
    """Return the list of tickers currently in the store."""
    return list(_data.keys())


# ── Initialisation (called once at metrics-agent startup) ─────────────────────

def initialize_all(tickers: list[str] | None = None) -> list[str]:
    """Pre-load balance sheet metrics for all tickers from disk.

    Should be called once during metrics-agent initialisation so that
    every subsequent get_metrics() call is a pure dict lookup with no I/O.

    Args:
        tickers: List of tickers to load. If None, all known companies are used.

    Returns:
        List of tickers successfully loaded.
    """
    global _initialized
    if _initialized:
        return loaded_tickers()

    if tickers is None:
        from config.companies import COMPANIES
        tickers = [c["ticker"] for c in COMPANIES]

    newly_loaded: list[str] = []
    missing: list[str] = []

    for ticker in tickers:
        if ticker in _data:
            continue  # already populated by orchestrator in this process
        if _try_load_from_disk(ticker):
            newly_loaded.append(ticker)
        else:
            missing.append(ticker)

    _initialized = True

    if newly_loaded:
        logger.info(f"[BSStore] Initialized from disk for: {newly_loaded}")
    if missing:
        for t in missing:
            logger.info(
                f"[BSStore] Balance sheet section not found in SEC parsing for {t}"
            )

    logger.info(
        f"[BSStore] Ready — {len(_data)} tickers in store: {list(_data.keys())}"
    )
    return list(_data.keys())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_load_from_disk(ticker: str) -> bool:
    """Load one ticker's balance sheet metrics from the saved JSON file.

    Returns True on success, False if the file is missing or unreadable.
    """
    path = _BS_DIR / ticker / "balance_sheet_metrics.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            _data[ticker] = json.load(f)
        logger.debug(f"[BSStore] Loaded from disk: {ticker}")
        return True
    except Exception as e:
        logger.warning(f"[BSStore] Failed to load from disk for {ticker}: {e}")
        return False
