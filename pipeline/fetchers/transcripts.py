"""Fetch earnings call transcripts.

Primary: earningscall library (full transcripts, free for AAPL/MSFT; paid for others).
Fallback: SEC EDGAR 8-K EX-99.1 earnings press releases (free for all companies).
"""

import asyncio
import logging
from datetime import datetime

from earningscall import get_company as _ec_get_company
from earningscall.errors import InsufficientApiAccessError

from pipeline.fetchers.earnings_releases import EarningsReleaseFetcher

logger = logging.getLogger(__name__)

_earnings_release_fetcher = EarningsReleaseFetcher()


def _fetch_transcripts_sync(ticker: str, num_quarters: int) -> list[dict] | None:
    """Try earningscall. Returns None if the ticker requires a paid API key."""
    try:
        company = _ec_get_company(ticker)
        if not company:
            return None
    except InsufficientApiAccessError:
        return None

    results = []
    now_ts = datetime.now().timestamp()

    for event in company.events():
        if len(results) >= num_quarters:
            break
        if event.conference_date and now_ts < event.conference_date.timestamp():
            continue
        try:
            transcript = company.get_transcript(event=event)
        except InsufficientApiAccessError:
            return None  # Requires paid key — signal fallback

        if not transcript or not transcript.text:
            continue

        results.append({
            "ticker": ticker,
            "year": event.year,
            "quarter": event.quarter,
            "date": (
                event.conference_date.strftime("%Y-%m-%d")
                if event.conference_date else ""
            ),
            "content": transcript.text,
        })
        logger.info(
            f"{ticker}: fetched Q{event.quarter} {event.year} transcript "
            f"({len(transcript.text):,} chars) via earningscall"
        )

    return results


class TranscriptFetcher:
    """Fetches earnings call transcripts with automatic source selection.

    - earningscall library → full transcripts (free for select tickers)
    - SEC EDGAR EX-99.1    → earnings press releases (free for all companies)
    """

    async def fetch_company_transcripts(
        self, ticker: str, cik: str = "", num_quarters: int = 4
    ) -> list[dict]:
        # Try earningscall first (non-blocking via thread)
        results = await asyncio.to_thread(
            _fetch_transcripts_sync, ticker, num_quarters
        )

        if results is not None:
            logger.info(
                f"{ticker}: {len(results)} transcripts via earningscall"
            )
            return results

        # Fallback: EDGAR earnings press releases
        logger.info(
            f"{ticker}: earningscall requires paid key — "
            f"falling back to SEC EDGAR earnings releases"
        )
        if not cik:
            logger.warning(f"{ticker}: no CIK provided for EDGAR fallback")
            return []

        releases = await _earnings_release_fetcher.fetch_company_earnings_releases(
            ticker, cik, num_releases=num_quarters
        )
        # Tag as earnings_release so parsers/agents know the source
        for r in releases:
            r["source"] = "edgar_earnings_release"
        return releases
