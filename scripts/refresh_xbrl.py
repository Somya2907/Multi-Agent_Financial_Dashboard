"""Re-parse cached XBRL companyfacts responses into data/financials/*.json.

HTTP responses are already cached at data/raw/xbrl/companyfacts_{TICKER}.json,
so this runs entirely from disk and takes a few seconds for all 50 companies.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.companies import COMPANIES  # noqa: E402
from pipeline.fetchers.xbrl_fetcher import XBRLFetcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("refresh_xbrl")

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "financials"


async def main() -> None:
    fetcher = XBRLFetcher()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, missing = 0, 0
    for company in COMPANIES:
        ticker, cik = company["ticker"], company["cik"]
        bs = await fetcher.fetch_balance_sheet(ticker, cik)
        if not bs:
            missing += 1
            logger.warning(f"{ticker}: nothing extracted")
            continue

        path = OUT_DIR / f"{ticker}_balance_sheet.json"
        with open(path, "w") as f:
            json.dump(bs, f, indent=2)
        ok += 1

    print(f"\nDone. {ok} written, {missing} missing → {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
