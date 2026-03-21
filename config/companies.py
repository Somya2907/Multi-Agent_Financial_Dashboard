"""Top 50 US public companies by market cap with SEC EDGAR CIK mappings."""

COMPANIES = [
    # ── Technology ────────────────────────────────────────────────────────────
    {"ticker": "AAPL",  "cik": "0000320193", "name": "Apple Inc.",                      "sector": "Technology"},
    {"ticker": "MSFT",  "cik": "0000789019", "name": "Microsoft Corporation",            "sector": "Technology"},
    {"ticker": "NVDA",  "cik": "0001045810", "name": "NVIDIA Corporation",               "sector": "Technology"},
    {"ticker": "AMZN",  "cik": "0001018724", "name": "Amazon.com Inc.",                  "sector": "Technology"},
    {"ticker": "GOOGL", "cik": "0001652044", "name": "Alphabet Inc.",                    "sector": "Technology"},
    {"ticker": "META",  "cik": "0001326801", "name": "Meta Platforms Inc.",              "sector": "Technology"},
    {"ticker": "AVGO",  "cik": "0001730168", "name": "Broadcom Inc.",                    "sector": "Technology"},
    {"ticker": "TSLA",  "cik": "0001318605", "name": "Tesla Inc.",                       "sector": "Technology"},
    {"ticker": "ORCL",  "cik": "0001341439", "name": "Oracle Corporation",               "sector": "Technology"},
    {"ticker": "CRM",   "cik": "0001108524", "name": "Salesforce Inc.",                  "sector": "Technology"},
    {"ticker": "AMD",   "cik": "0000002488", "name": "Advanced Micro Devices Inc.",      "sector": "Technology"},
    {"ticker": "ADBE",  "cik": "0000796343", "name": "Adobe Inc.",                       "sector": "Technology"},
    {"ticker": "CSCO",  "cik": "0000858877", "name": "Cisco Systems Inc.",               "sector": "Technology"},
    {"ticker": "TXN",   "cik": "0000097476", "name": "Texas Instruments Inc.",           "sector": "Technology"},
    {"ticker": "INTC",  "cik": "0000050863", "name": "Intel Corporation",                "sector": "Technology"},
    {"ticker": "QCOM",  "cik": "0000804328", "name": "Qualcomm Inc.",                    "sector": "Technology"},
    {"ticker": "INTU",  "cik": "0000896878", "name": "Intuit Inc.",                      "sector": "Technology"},
    {"ticker": "IBM",   "cik": "0000051143", "name": "International Business Machines",  "sector": "Technology"},
    {"ticker": "NFLX",  "cik": "0001065280", "name": "Netflix Inc.",                     "sector": "Technology"},

    # ── Financial Services ────────────────────────────────────────────────────
    {"ticker": "BRK-B", "cik": "0001067983", "name": "Berkshire Hathaway Inc.",          "sector": "Financial Services"},
    {"ticker": "JPM",   "cik": "0000019617", "name": "JPMorgan Chase & Co.",             "sector": "Financial Services"},
    {"ticker": "V",     "cik": "0001403161", "name": "Visa Inc.",                        "sector": "Financial Services"},
    {"ticker": "MA",    "cik": "0001141391", "name": "Mastercard Inc.",                  "sector": "Financial Services"},
    {"ticker": "BAC",   "cik": "0000070858", "name": "Bank of America Corporation",      "sector": "Financial Services"},
    {"ticker": "WFC",   "cik": "0000072971", "name": "Wells Fargo & Company",            "sector": "Financial Services"},

    # ── Healthcare ────────────────────────────────────────────────────────────
    {"ticker": "LLY",   "cik": "0000059478", "name": "Eli Lilly and Company",            "sector": "Healthcare"},
    {"ticker": "UNH",   "cik": "0000731766", "name": "UnitedHealth Group Inc.",          "sector": "Healthcare"},
    {"ticker": "JNJ",   "cik": "0000200406", "name": "Johnson & Johnson",                "sector": "Healthcare"},
    {"ticker": "ABBV",  "cik": "0001551152", "name": "AbbVie Inc.",                      "sector": "Healthcare"},
    {"ticker": "MRK",   "cik": "0000310158", "name": "Merck & Co. Inc.",                 "sector": "Healthcare"},
    {"ticker": "TMO",   "cik": "0000097745", "name": "Thermo Fisher Scientific Inc.",    "sector": "Healthcare"},
    {"ticker": "ABT",   "cik": "0000001800", "name": "Abbott Laboratories",              "sector": "Healthcare"},
    {"ticker": "DHR",   "cik": "0000313616", "name": "Danaher Corporation",              "sector": "Healthcare"},

    # ── Consumer ──────────────────────────────────────────────────────────────
    {"ticker": "WMT",   "cik": "0000104169", "name": "Walmart Inc.",                     "sector": "Consumer Staples"},
    {"ticker": "COST",  "cik": "0000909832", "name": "Costco Wholesale Corporation",     "sector": "Consumer Staples"},
    {"ticker": "PG",    "cik": "0000080424", "name": "Procter & Gamble Company",         "sector": "Consumer Staples"},
    {"ticker": "KO",    "cik": "0000021344", "name": "The Coca-Cola Company",            "sector": "Consumer Staples"},
    {"ticker": "PEP",   "cik": "0000077476", "name": "PepsiCo Inc.",                     "sector": "Consumer Staples"},
    {"ticker": "MCD",   "cik": "0000063908", "name": "McDonald's Corporation",           "sector": "Consumer Discretionary"},
    {"ticker": "NKE",   "cik": "0000320187", "name": "Nike Inc.",                        "sector": "Consumer Discretionary"},
    {"ticker": "PM",    "cik": "0001413329", "name": "Philip Morris International",      "sector": "Consumer Staples"},

    # ── Energy ────────────────────────────────────────────────────────────────
    {"ticker": "XOM",   "cik": "0000034088", "name": "Exxon Mobil Corporation",          "sector": "Energy"},
    {"ticker": "CVX",   "cik": "0000093410", "name": "Chevron Corporation",              "sector": "Energy"},

    # ── Industrial ────────────────────────────────────────────────────────────
    {"ticker": "CAT",   "cik": "0000018230", "name": "Caterpillar Inc.",                 "sector": "Industrial"},
    {"ticker": "GE",    "cik": "0000040545", "name": "GE Aerospace",                     "sector": "Industrial"},
    {"ticker": "UPS",   "cik": "0001090727", "name": "United Parcel Service Inc.",       "sector": "Industrial"},
    {"ticker": "HD",    "cik": "0000354950", "name": "The Home Depot Inc.",              "sector": "Industrial"},

    # ── Other ─────────────────────────────────────────────────────────────────
    {"ticker": "ACN",   "cik": "0001467373", "name": "Accenture plc",                    "sector": "Technology"},
    {"ticker": "LIN",   "cik": "0001707925", "name": "Linde plc",                        "sector": "Industrial"},
    {"ticker": "NEE",   "cik": "0000753308", "name": "NextEra Energy Inc.",              "sector": "Utilities"},
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
