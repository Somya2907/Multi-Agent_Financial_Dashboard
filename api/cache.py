"""File-based cache for company analysis results.

Stores each ticker's full GraphState output as JSON at data/cache/{ticker}.json.
TTL is enforced by file mtime; stale entries are ignored and regenerated.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cached analyses are valid for 24 hours — SEC filings don't change intraday
_TTL_SECONDS = 24 * 60 * 60


def _path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}.json"


def get_cached(ticker: str) -> dict | None:
    """Return cached analysis for ticker, or None if missing/stale."""
    path = _path(ticker)
    if not path.exists():
        return None

    age = time.time() - path.stat().st_mtime
    if age > _TTL_SECONDS:
        logger.info(f"[Cache] {ticker} stale ({age/3600:.1f}h old) — will regenerate")
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"[Cache] HIT {ticker} ({age/60:.0f}m old)")
        return data
    except Exception as e:
        logger.warning(f"[Cache] Failed to read {path}: {e}")
        return None


def save_cached(ticker: str, result: dict) -> None:
    """Persist analysis result for ticker."""
    path = _path(ticker)
    try:
        with open(path, "w") as f:
            json.dump(result, f, default=str)
        logger.info(f"[Cache] SAVED {ticker} → {path}")
    except Exception as e:
        logger.warning(f"[Cache] Failed to write {path}: {e}")


def invalidate(ticker: str) -> bool:
    """Delete cached entry for ticker. Returns True if a file was removed."""
    path = _path(ticker)
    if path.exists():
        path.unlink()
        logger.info(f"[Cache] INVALIDATED {ticker}")
        return True
    return False
