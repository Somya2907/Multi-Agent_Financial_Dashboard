"""Pipeline orchestrator: coordinates fetch → parse → chunk → embed → index."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from config.companies import COMPANIES, get_all_tickers
from pipeline.fetchers.sec_edgar import SECEdgarFetcher
from pipeline.fetchers.transcripts import TranscriptFetcher
from pipeline.fetchers.xbrl_fetcher import XBRLFetcher
from pipeline.parsers.sec_parser import parse_filing
from pipeline.parsers.transcript_parser import parse_transcript
from pipeline.chunking.chunker import chunk_document
from pipeline.embedding.embedder import TitanEmbedder
from pipeline.indexing.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks"


async def _fetch_all_data(
    tickers: list[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Fetch all raw data from APIs.

    Returns:
        (sec_filings, transcripts, xbrl_balance_sheets_by_ticker)
    """
    companies = COMPANIES
    if tickers:
        companies = [c for c in COMPANIES if c["ticker"] in tickers]

    sec_fetcher = SECEdgarFetcher()
    transcript_fetcher = TranscriptFetcher()
    xbrl_fetcher = XBRLFetcher()

    all_sec_filings = []
    all_transcripts = []
    xbrl_balance_sheets = {}

    for company in companies:
        ticker = company["ticker"]
        cik = company["cik"]
        logger.info(f"--- Fetching data for {ticker} ---")

        # SEC 10-K / 10-Q filings (HTML → FAISS qualitative index)
        filings = await sec_fetcher.fetch_company_filings(ticker, cik)
        all_sec_filings.extend(filings)

        # Earnings call transcripts (earningscall → EDGAR press release fallback)
        transcripts = await transcript_fetcher.fetch_company_transcripts(ticker, cik=cik)
        all_transcripts.extend(transcripts)

        # Structured balance sheet via SEC EDGAR XBRL companyfacts API
        bs = await xbrl_fetcher.fetch_balance_sheet(ticker, cik)
        if bs:
            xbrl_balance_sheets[ticker] = bs

    return all_sec_filings, all_transcripts, xbrl_balance_sheets


def _parse_all_data(
    sec_filings: list[dict],
    transcripts: list[dict],
) -> list[dict]:
    """Parse all raw data into section-segmented documents for the FAISS index.

    Balance sheet metrics are sourced exclusively from the XBRL API
    (see _save_xbrl_balance_sheets).  HTML filing text is parsed here so
    qualitative sections (Risk Factors, MD&A, …) can be chunked and embedded.
    """
    all_documents = []

    # Parse SEC filings → qualitative text for FAISS
    for filing in sec_filings:
        sections = parse_filing(
            html=filing["html"],
            ticker=filing["ticker"],
            form_type=filing["form"],
            filing_date=filing["filingDate"],
            source_url=filing.get("source_url", ""),
        )
        all_documents.extend(sections)

    # Parse transcripts
    for t in transcripts:
        sections = parse_transcript(
            content=t["content"],
            ticker=t["ticker"],
            year=t["year"],
            quarter=t["quarter"],
            date=t.get("date", ""),
        )
        all_documents.extend(sections)

    logger.info(f"Total parsed documents/sections: {len(all_documents)}")
    return all_documents


def _chunk_all_documents(documents: list[dict]) -> list[dict]:
    """Chunk all parsed documents."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def _save_chunks(chunks: list[dict]):
    """Save chunks to JSONL files organized by ticker."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    by_ticker = {}
    for chunk in chunks:
        ticker = chunk.get("ticker", "unknown")
        by_ticker.setdefault(ticker, []).append(chunk)

    for ticker, ticker_chunks in by_ticker.items():
        ticker_dir = CHUNKS_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        path = ticker_dir / "all_chunks.jsonl"
        with open(path, "w") as f:
            for chunk in ticker_chunks:
                f.write(json.dumps(chunk) + "\n")
        logger.info(f"Saved {len(ticker_chunks)} chunks for {ticker}")


def _save_xbrl_balance_sheets(xbrl_by_ticker: dict):
    """Write XBRL-derived balance sheet values to data/financials/{ticker}_balance_sheet.json.

    This is the canonical balance sheet source consumed by the financial
    metrics agent at query time.  Format::

        {
            "current_assets":       143566000000.0,
            "current_liabilities":  134690000000.0,
            "total_assets":         364980000000.0,
            "total_liabilities":    308030000000.0,
            "shareholder_equity":    56950000000.0
        }
    """
    out_dir = Path(__file__).resolve().parents[1] / "data" / "financials"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker, values in xbrl_by_ticker.items():
        if not values:
            continue
        path = out_dir / f"{ticker}_balance_sheet.json"
        with open(path, "w") as f:
            json.dump(values, f, indent=2)
        logger.info(
            f"[XBRL] Saved balance sheet for {ticker}: "
            + ", ".join(f"{k}={v:,.0f}" for k, v in values.items())
        )


async def run_pipeline(
    tickers: list[str] | None = None,
    skip_embedding: bool = False,
):
    """Run the full data pipeline: fetch → parse → chunk → embed → index.

    Args:
        tickers: List of tickers to process. None = all companies.
        skip_embedding: If True, skip embedding and indexing (useful for testing).
    """
    logger.info("=== Starting Data Pipeline ===")

    # Step 1: Fetch
    logger.info("Step 1: Fetching data from APIs...")
    sec_filings, transcripts, xbrl_bs = await _fetch_all_data(tickers)
    _save_xbrl_balance_sheets(xbrl_bs)         # → data/financials/{ticker}_balance_sheet.json
    logger.info(
        f"Fetched: {len(sec_filings)} filings, {len(transcripts)} transcripts, "
        f"XBRL balance sheets for: {list(xbrl_bs.keys())}"
    )

    # Step 2: Parse (qualitative text for FAISS)
    logger.info("Step 2: Parsing documents...")
    documents = _parse_all_data(sec_filings, transcripts)

    # Step 3: Chunk
    logger.info("Step 3: Chunking documents...")
    chunks = _chunk_all_documents(documents)
    _save_chunks(chunks)

    if skip_embedding:
        logger.info("Skipping embedding and indexing (skip_embedding=True)")
        return chunks

    # Step 4: Embed
    logger.info("Step 4: Embedding chunks...")
    embedder = TitanEmbedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_batch(texts)

    # Step 5: Index
    logger.info("Step 5: Building FAISS index...")
    manager = FAISSManager()
    manager.build_index(vectors, chunks)
    manager.save()

    logger.info(
        f"=== Pipeline Complete: {len(chunks)} chunks indexed ==="
    )
    return chunks
