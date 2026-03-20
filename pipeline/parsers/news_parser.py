"""Parse financial news articles into documents for chunking."""

import logging

logger = logging.getLogger(__name__)


def parse_news_articles(articles: list[dict]) -> list[dict]:
    """Parse a list of news articles into documents ready for chunking.

    Each article becomes a single document with title + text concatenated.
    """
    results = []
    for article in articles:
        title = article.get("title", "").strip()
        text = article.get("text", "").strip()
        if not text and not title:
            continue

        full_text = f"{title}\n\n{text}" if title and text else (title or text)

        results.append(
            {
                "ticker": article.get("ticker", ""),
                "source_type": "news",
                "filing_date": article.get("publishedDate", ""),
                "section_name": "News Article",
                "text": full_text,
                "source_url": article.get("url", ""),
            }
        )

    logger.info(f"Parsed {len(results)} news articles")
    return results
