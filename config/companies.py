"""Top US public companies by market cap. Starting with 5 for development."""

COMPANIES = [
    {
        "ticker": "AAPL",
        "cik": "0000320193",
        "name": "Apple Inc.",
        "sector": "Technology",
    },
    {
        "ticker": "MSFT",
        "cik": "0000789019",
        "name": "Microsoft Corporation",
        "sector": "Technology",
    },
    {
        "ticker": "NVDA",
        "cik": "0001045810",
        "name": "NVIDIA Corporation",
        "sector": "Technology",
    },
    {
        "ticker": "JPM",
        "cik": "0000019617",
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial Services",
    },
    {
        "ticker": "JNJ",
        "cik": "0000200406",
        "name": "Johnson & Johnson",
        "sector": "Healthcare",
    },
]


def get_company(ticker: str) -> dict | None:
    """Look up a company by ticker symbol."""
    for company in COMPANIES:
        if company["ticker"] == ticker:
            return company
    return None


def get_all_tickers() -> list[str]:
    """Return all ticker symbols."""
    return [c["ticker"] for c in COMPANIES]
