"""Fetch earnings press releases (EX-99.1) from SEC EDGAR 8-K Item 2.02 filings.

Companies file quarterly earnings results as Form 8-K with Item 2.02
(Results of Operations and Financial Condition). The EX-99.1 exhibit contains
the full press release: CEO/CFO quotes, segment revenue, YoY comparisons,
forward guidance, and risk statements — a useful substitute for call transcripts.
"""

import logging
import re

from bs4 import BeautifulSoup

from config.settings import settings
from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


class EarningsReleaseFetcher(BaseFetcher):
    """Fetches earnings press releases from EDGAR 8-K Item 2.02 filings."""

    def __init__(self):
        super().__init__(
            source_name="earnings_releases",
            rate_limit_seconds=settings.sec_rate_limit,
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )

    async def _fetch_submissions(self, cik: str) -> dict | None:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        return await self.fetch(url, cache_key=f"{cik}_submissions", response_type="json")

    def _find_earnings_8ks(self, submissions: dict, max_filings: int = 6) -> list[dict]:
        """Return 8-K filings where items contains '2.02' (earnings releases)."""
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        items_list = recent.get("items", [])

        results = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            items = items_list[i] if i < len(items_list) else ""
            if "2.02" not in str(items):
                continue
            results.append({
                "accessionNumber": accessions[i],
                "filingDate": dates[i],
            })
            if len(results) >= max_filings:
                break

        return results

    async def _fetch_filing_index(
        self, cik: str, accession: str
    ) -> list[tuple[str, str, str]]:
        """Fetch filing index HTML and return list of (filename, description, type)."""
        cik_raw = cik.lstrip("0")
        acc_no_dashes = accession.replace("-", "")
        index_url = (
            f"{SEC_ARCHIVES}/{cik_raw}/{acc_no_dashes}/{accession}-index.htm"
        )
        html = await self.fetch(
            index_url,
            cache_key=f"{cik_raw}_{acc_no_dashes}_index",
            response_type="text",
        )
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        docs = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            # EDGAR index table columns: [seq, description, filename, type, size]
            if len(cells) < 4:
                continue
            desc = cells[1].get_text(strip=True)
            link = cells[2].find("a")
            fname = link.get_text(strip=True) if link else cells[2].get_text(strip=True)
            doc_type = cells[3].get_text(strip=True)
            if fname:
                docs.append((fname, desc, doc_type))

        return docs

    def _find_ex99_doc(self, docs: list[tuple[str, str, str]]) -> str | None:
        """Find the EX-99.1 exhibit filename from the filing document list."""
        for fname, desc, doc_type in docs:
            if "EX-99.1" in doc_type or "EX-99" in doc_type:
                if fname.endswith((".htm", ".html", ".txt")):
                    return fname
        # Fallback: look for filename containing 'ex99' or 'exhibit99'
        for fname, desc, doc_type in docs:
            if re.search(r"ex[-_]?99", fname, re.IGNORECASE):
                return fname
        return None

    async def fetch_company_earnings_releases(
        self, ticker: str, cik: str, num_releases: int = 4
    ) -> list[dict]:
        """Fetch recent earnings press releases for a company.

        Returns list of dicts with ticker, filing_date, year, quarter, content.
        """
        submissions = await self._fetch_submissions(cik)
        if not submissions:
            logger.error(f"No submissions for {ticker}")
            return []

        filings = self._find_earnings_8ks(submissions, max_filings=num_releases + 4)
        logger.info(f"{ticker}: found {len(filings)} earnings 8-K filings")

        cik_raw = cik.lstrip("0")
        results = []

        for filing in filings:
            if len(results) >= num_releases:
                break

            acc = filing["accessionNumber"]
            date = filing["filingDate"]
            acc_no_dashes = acc.replace("-", "")

            # Get filing index to find EX-99.1
            docs = await self._fetch_filing_index(cik, acc)
            ex99_fname = self._find_ex99_doc(docs)

            if not ex99_fname:
                logger.debug(f"{ticker} {date}: no EX-99.1 found in 8-K {acc}")
                continue

            # Fetch the press release document
            doc_url = f"{SEC_ARCHIVES}/{cik_raw}/{acc_no_dashes}/{ex99_fname}"
            html = await self.fetch(
                doc_url,
                cache_key=f"{cik_raw}_{acc_no_dashes}_ex991",
                response_type="text",
            )
            if not html:
                continue

            # Extract plain text
            soup = BeautifulSoup(html, "lxml")
            for tag in soup.find_all(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if len(text) < 200:
                continue

            # Infer quarter from filing date
            month = int(date[5:7]) if date else 0
            quarter = (month - 1) // 3 + 1
            year = int(date[:4]) if date else 0

            results.append({
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "date": date,
                "content": text,
                "source_url": doc_url,
            })
            logger.info(f"{ticker}: fetched earnings release Q{quarter} {year}")

        logger.info(f"{ticker}: {len(results)} earnings releases fetched")
        return results
