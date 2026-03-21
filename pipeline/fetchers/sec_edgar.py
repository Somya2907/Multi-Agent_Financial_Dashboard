"""SEC EDGAR API fetcher for 10-K, 10-Q, and 8-K filings."""

import logging
import re

from config.settings import settings
from pipeline.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accession_no_dashes}/{document}"
)


class SECEdgarFetcher(BaseFetcher):
    """Fetches SEC 10-K, 10-Q, and 8-K filings via the EDGAR API."""

    def __init__(self):
        super().__init__(
            source_name="sec_filings",
            rate_limit_seconds=settings.sec_rate_limit,
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )

    async def get_submissions(self, cik: str) -> dict | None:
        """Fetch the submissions JSON for a company (contains filing metadata)."""
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        cache_key = f"{cik}_submissions"
        return await self.fetch(url, cache_key=cache_key, response_type="json")

    def _extract_filings(
        self,
        submissions: dict,
        form_types: list[str],
        max_filings: int = 5,
    ) -> list[dict]:
        """Extract filing metadata from the submissions JSON.

        Returns a list of dicts with: accessionNumber, filingDate, primaryDocument, form.
        """
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        documents = recent.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form in form_types and i < len(accessions):
                filings.append(
                    {
                        "form": form,
                        "accessionNumber": accessions[i],
                        "filingDate": dates[i] if i < len(dates) else "",
                        "primaryDocument": documents[i] if i < len(documents) else "",
                    }
                )
                if len(filings) >= max_filings:
                    break

        return filings

    def _build_filing_url(self, cik: str, filing: dict) -> str:
        """Build the URL to fetch the actual filing document."""
        cik_raw = cik.lstrip("0")
        accession_no_dashes = filing["accessionNumber"].replace("-", "")
        return SEC_ARCHIVES_URL.format(
            cik_raw=cik_raw,
            accession_no_dashes=accession_no_dashes,
            document=filing["primaryDocument"],
        )

    async def fetch_filing_document(
        self, cik: str, ticker: str, filing: dict
    ) -> str | None:
        """Fetch the HTML content of a single filing."""
        url = self._build_filing_url(cik, filing)
        cache_key = (
            f"{ticker}/{filing['form']}_{filing['filingDate']}"
        )
        return await self.fetch(url, cache_key=cache_key, response_type="text")

    async def fetch_company_filings(
        self,
        ticker: str,
        cik: str,
        num_10k: int = 1,
        num_10q: int = 4,
        num_8k: int = 8,
    ) -> list[dict]:
        """Fetch all target filings for a company.

        Fetches 10-K (annual), 10-Q (quarterly), and 8-K (current reports /
        material events) filings.  8-K primary documents cover material events
        such as acquisitions, executive changes, restructurings, and earnings
        disclosures — useful context for qualitative and risk queries.

        Returns a list of dicts with keys: ticker, form, filingDate, accessionNumber, html.
        """
        submissions = await self.get_submissions(cik)
        if not submissions:
            logger.error(f"Failed to fetch submissions for {ticker} (CIK: {cik})")
            return []

        # Extract filing metadata for each form type
        filings_10k = self._extract_filings(submissions, ["10-K"], max_filings=num_10k)
        filings_10q = self._extract_filings(submissions, ["10-Q"], max_filings=num_10q)
        filings_8k  = self._extract_filings(submissions, ["8-K"],  max_filings=num_8k)
        all_filings = filings_10k + filings_10q + filings_8k

        logger.info(
            f"{ticker}: found {len(filings_10k)} 10-K, "
            f"{len(filings_10q)} 10-Q, {len(filings_8k)} 8-K filings"
        )

        results = []
        for filing in all_filings:
            html = await self.fetch_filing_document(cik, ticker, filing)
            if html:
                results.append(
                    {
                        "ticker": ticker,
                        "form": filing["form"],
                        "filingDate": filing["filingDate"],
                        "accessionNumber": filing["accessionNumber"],
                        "html": html,
                        "source_url": self._build_filing_url(cik, filing),
                    }
                )
            else:
                logger.warning(
                    f"Failed to fetch {filing['form']} for {ticker} "
                    f"(date: {filing['filingDate']})"
                )

        return results
