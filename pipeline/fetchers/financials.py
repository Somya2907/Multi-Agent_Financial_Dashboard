"""Fetch structured financial data via Financial Modeling Prep API."""

import logging

from config.settings import settings
from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# FMP endpoints for structured financial data (not chunked/embedded — used directly by agents)
FINANCIAL_ENDPOINTS = {
    "income_statement": "income-statement",
    "balance_sheet": "balance-sheet-statement",
    "cash_flow": "cash-flow-statement",
    "key_metrics": "key-metrics",
    "ratios": "ratios",
}


class FinancialsFetcher(BaseFetcher):
    """Fetches structured financial statements and ratios from FMP."""

    def __init__(self):
        super().__init__(
            source_name="financials",
            rate_limit_seconds=0.5,
            cache_dir="financials",
        )

    async def fetch_financial_data(
        self, ticker: str, endpoint_key: str, period: str = "annual", limit: int = 3
    ) -> list[dict] | None:
        """Fetch a single financial data endpoint."""
        endpoint = FINANCIAL_ENDPOINTS.get(endpoint_key)
        if not endpoint:
            logger.error(f"Unknown endpoint: {endpoint_key}")
            return None

        url = (
            f"{settings.fmp_base_url}/{endpoint}/{ticker}"
            f"?period={period}&limit={limit}&apikey={settings.fmp_api_key}"
        )
        cache_key = f"{ticker}/{endpoint_key}_{period}"
        return await self.fetch(url, cache_key=cache_key, response_type="json")

    async def fetch_all_financials(
        self, ticker: str, period: str = "annual", limit: int = 3
    ) -> dict[str, list[dict]]:
        """Fetch all financial data endpoints for a company.

        Returns a dict keyed by endpoint name, each containing a list of period records.
        This data is stored as JSON and read directly by the Financial Metrics Agent
        (not chunked or embedded).
        """
        results = {}
        for key in FINANCIAL_ENDPOINTS:
            data = await self.fetch_financial_data(ticker, key, period, limit)
            if data and isinstance(data, list):
                results[key] = data
            else:
                logger.warning(f"No {key} data for {ticker}")
                results[key] = []

        total = sum(len(v) for v in results.values())
        logger.info(f"{ticker}: fetched {total} financial records across {len(results)} endpoints")
        return results
