"""Fetch financial news via Financial Modeling Prep API."""

import logging

from config.settings import settings
from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class NewsFetcher(BaseFetcher):
    """Fetches financial news articles from FMP."""

    def __init__(self):
        super().__init__(
            source_name="news",
            rate_limit_seconds=0.5,
            cache_dir="news",
        )

    async def fetch_company_news(
        self, ticker: str, limit: int = 50
    ) -> list[dict]:
        """Fetch recent news articles for a company."""
        url = (
            f"{settings.fmp_base_url}/stock_news"
            f"?tickers={ticker}&limit={limit}&apikey={settings.fmp_api_key}"
        )
        cache_key = f"{ticker}/latest_news"
        data = await self.fetch(url, cache_key=cache_key, response_type="json")

        if not data or not isinstance(data, list):
            logger.warning(f"No news found for {ticker}")
            return []

        results = []
        for article in data:
            results.append(
                {
                    "ticker": ticker,
                    "title": article.get("title", ""),
                    "text": article.get("text", ""),
                    "publishedDate": article.get("publishedDate", ""),
                    "site": article.get("site", ""),
                    "url": article.get("url", ""),
                }
            )

        logger.info(f"{ticker}: fetched {len(results)} news articles")
        return results
