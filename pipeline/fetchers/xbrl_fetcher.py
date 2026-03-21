"""Fetch structured balance sheet values from SEC EDGAR's XBRL companyfacts API.

API endpoint:
    https://data.sec.gov/api/xbrl/companyfacts/CIK{10-digit-cik}.json

The response contains every XBRL-tagged financial fact reported across all
filings.  We filter to the five core balance sheet tags, keep only annual
10-K observations, and take the most recent end-date value.

GAAP tag → internal field mapping
----------------------------------
  AssetsCurrent       → current_assets
  LiabilitiesCurrent  → current_liabilities
  Assets              → total_assets
  Liabilities         → total_liabilities
  StockholdersEquity  → shareholder_equity
      (with fallback to the consolidated/NCI variant)

Values are in USD (dollars, not millions).
"""

import logging

from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

_XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# Ordered list: (internal_field, [gaap_tag, ...fallback_tags])
_TAG_MAP: list[tuple[str, list[str]]] = [
    ("current_assets",       ["AssetsCurrent"]),
    ("current_liabilities",  ["LiabilitiesCurrent"]),
    ("total_assets",         ["Assets"]),
    ("total_liabilities",    ["Liabilities"]),
    ("shareholder_equity",   [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
]


class XBRLFetcher(BaseFetcher):
    """Fetches and parses SEC EDGAR XBRL companyfacts for balance sheet data."""

    def __init__(self):
        super().__init__(
            source_name="xbrl",
            rate_limit_seconds=0.12,          # ≤ 10 req/s per SEC policy
            headers={"User-Agent": "Somaya Jain somayaj@andrew.cmu.edu"},
            cache_dir="xbrl",
        )

    async def fetch_balance_sheet(self, ticker: str, cik: str) -> dict[str, float] | None:
        """Return the most-recent annual balance sheet values for a ticker.

        Calls the XBRL companyfacts endpoint, extracts the five target fields,
        and returns a dict like::

            {
                "current_assets":       143_566_000_000.0,
                "current_liabilities":  134_690_000_000.0,
                "total_assets":         364_980_000_000.0,
                "total_liabilities":    308_030_000_000.0,
                "shareholder_equity":    56_950_000_000.0,
            }

        Returns None if the API call fails or no data is found.
        Values are raw USD (not scaled).
        """
        # Normalise CIK to 10 zero-padded digits
        cik_padded = str(int(cik)).zfill(10)
        url = f"{_XBRL_BASE}/CIK{cik_padded}.json"
        cache_key = f"companyfacts_{ticker}"

        logger.info(f"[XBRL] Fetching companyfacts for {ticker} (CIK {cik_padded})")
        data = await self.fetch(url, cache_key=cache_key)
        if not data:
            logger.warning(f"[XBRL] No companyfacts data returned for {ticker}")
            return None

        us_gaap: dict = data.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            logger.warning(f"[XBRL] No us-gaap facts for {ticker}")
            return None

        values: dict[str, float] = {}

        for field, tags in _TAG_MAP:
            for tag in tags:
                tag_data = us_gaap.get(tag)
                if not tag_data:
                    continue

                usd_entries: list[dict] = tag_data.get("units", {}).get("USD", [])
                if not usd_entries:
                    logger.debug(f"[XBRL] {ticker}: {tag} has no USD unit entries")
                    continue

                # Prefer 10-K annual filings; fall back to any filing
                annual = [e for e in usd_entries if e.get("form") == "10-K"]
                candidates = annual if annual else usd_entries

                # Most recent by period end date
                best = max(candidates, key=lambda e: e.get("end", ""))
                val = best.get("val")
                if val is None:
                    logger.debug(
                        f"[XBRL] {ticker}: {tag} most-recent entry has no 'val'"
                    )
                    continue

                values[field] = float(val)
                logger.info(
                    f"[XBRL] {ticker}: {field} = {val:,.0f} "
                    f"(tag={tag}, end={best.get('end')})"
                )
                break  # first matching tag wins; skip fallbacks

            if field not in values:
                logger.warning(
                    f"[XBRL] {ticker}: could not extract {field} "
                    f"(tried: {tags})"
                )

        if not values:
            logger.warning(f"[XBRL] {ticker}: no balance sheet fields extracted")
            return None

        logger.info(
            f"[XBRL] {ticker}: extracted {len(values)}/5 fields — "
            + ", ".join(f"{k}={v:,.0f}" for k, v in values.items())
        )
        return values
