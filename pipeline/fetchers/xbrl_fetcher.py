"""Fetch structured financial statement values from SEC EDGAR's XBRL companyfacts API.

API endpoint:
    https://data.sec.gov/api/xbrl/companyfacts/CIK{10-digit-cik}.json

The response contains every XBRL-tagged financial fact reported across all
filings.  We filter to the target tags, keep only annual 10-K observations,
and take the most recent end-date value.

Coverage: balance sheet, income statement, cash flow statement (most-recent
fiscal year only — TTM and historical series are out of scope).

Values are in USD unless noted otherwise (EPS in USD/shares).
"""

import logging

from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

_XBRL_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

# Each entry: (internal_field, [gaap_tag, ...fallbacks], unit)
# Fallback tags let us handle ASC 606 / legacy filers / variations in cash tagging.
_TAG_MAP: list[tuple[str, list[str], str]] = [
    # ── Balance sheet ────────────────────────────────────────────────────────
    ("current_assets",            ["AssetsCurrent"], "USD"),
    ("current_liabilities",       ["LiabilitiesCurrent"], "USD"),
    ("total_assets",              ["Assets"], "USD"),
    ("total_liabilities",         ["Liabilities"], "USD"),
    ("shareholder_equity", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ], "USD"),
    ("cash_and_equivalents", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ], "USD"),
    ("long_term_debt", [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ], "USD"),

    # ── Income statement ─────────────────────────────────────────────────────
    ("revenue", [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ], "USD"),
    ("cost_of_revenue", [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ], "USD"),
    ("gross_profit",       ["GrossProfit"], "USD"),
    ("operating_income",   ["OperatingIncomeLoss"], "USD"),
    ("net_income", [
        "NetIncomeLoss",
        "ProfitLoss",
    ], "USD"),
    ("eps_diluted", [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ], "USD/shares"),

    # ── Cash flow ────────────────────────────────────────────────────────────
    ("operating_cash_flow", ["NetCashProvidedByUsedInOperatingActivities"], "USD"),
    ("capex",               ["PaymentsToAcquirePropertyPlantAndEquipment"], "USD"),
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
        """Return the most-recent annual financial statement values for a ticker.

        Extracts balance sheet, income statement, and cash flow line items
        (15 fields total). Returns None if the API call fails or no data found.
        Monetary values are in USD; EPS is USD/shares.
        """
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

        for field, tags, unit in _TAG_MAP:
            # Collect candidate entries from every fallback tag so we can
            # pick the globally most-recent value (companies switch tags over
            # time, e.g. ASC 606 moved revenue from `Revenues` to
            # `RevenueFromContractWithCustomerExcludingAssessedTax`).
            all_candidates: list[tuple[dict, str]] = []
            for tag in tags:
                tag_data = us_gaap.get(tag)
                if not tag_data:
                    continue
                unit_entries: list[dict] = tag_data.get("units", {}).get(unit, [])
                if not unit_entries:
                    continue
                annual = [e for e in unit_entries if e.get("form") == "10-K"]
                for e in (annual if annual else unit_entries):
                    if e.get("val") is not None and e.get("end"):
                        all_candidates.append((e, tag))

            if not all_candidates:
                logger.debug(f"[XBRL] {ticker}: no value for {field} (tried: {tags})")
                continue

            best, best_tag = max(all_candidates, key=lambda pair: pair[0]["end"])
            values[field] = float(best["val"])
            logger.debug(
                f"[XBRL] {ticker}: {field} = {best['val']} "
                f"(tag={best_tag}, unit={unit}, end={best['end']})"
            )

        # Derive gross_profit if missing but revenue + cost_of_revenue available
        if "gross_profit" not in values and {"revenue", "cost_of_revenue"} <= values.keys():
            values["gross_profit"] = values["revenue"] - values["cost_of_revenue"]
            logger.debug(f"[XBRL] {ticker}: gross_profit derived from revenue - cost_of_revenue")

        if not values:
            logger.warning(f"[XBRL] {ticker}: no fields extracted")
            return None

        logger.info(f"[XBRL] {ticker}: extracted {len(values)}/{len(_TAG_MAP)} fields")
        return values
