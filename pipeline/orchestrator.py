"""Pipeline orchestrator: coordinates fetch → parse → chunk → embed → index."""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from config.companies import COMPANIES, get_all_tickers
from pipeline.fetchers.sec_edgar import SECEdgarFetcher
from pipeline.fetchers.transcripts import TranscriptFetcher
from pipeline.fetchers.news import NewsFetcher
from pipeline.fetchers.financials import FinancialsFetcher
from pipeline.parsers.sec_parser import parse_filing
from pipeline.parsers.transcript_parser import parse_transcript
from pipeline.parsers.news_parser import parse_news_articles
from pipeline.chunking.chunker import chunk_document
from pipeline.embedding.embedder import TitanEmbedder
from pipeline.indexing.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks"


async def _fetch_all_data(
    tickers: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Fetch all raw data from APIs.

    Returns: (sec_filings, transcripts, news_articles, financials_by_ticker)
    """
    companies = COMPANIES
    if tickers:
        companies = [c for c in COMPANIES if c["ticker"] in tickers]

    sec_fetcher = SECEdgarFetcher()
    transcript_fetcher = TranscriptFetcher()
    news_fetcher = NewsFetcher()
    financials_fetcher = FinancialsFetcher()

    all_sec_filings = []
    all_transcripts = []
    all_news = []
    all_financials = {}

    for company in companies:
        ticker = company["ticker"]
        cik = company["cik"]
        logger.info(f"--- Fetching data for {ticker} ---")

        # SEC 10-K / 10-Q filings
        filings = await sec_fetcher.fetch_company_filings(ticker, cik)
        all_sec_filings.extend(filings)

        # Earnings call transcripts (earningscall → EDGAR press release fallback)
        transcripts = await transcript_fetcher.fetch_company_transcripts(ticker, cik=cik)
        all_transcripts.extend(transcripts)

        # News
        news = await news_fetcher.fetch_company_news(ticker)
        all_news.extend(news)

        # Structured financials (stored separately, not chunked)
        financials = await financials_fetcher.fetch_all_financials(ticker)
        all_financials[ticker] = financials

    return all_sec_filings, all_transcripts, all_news, all_financials


def _parse_all_data(
    sec_filings: list[dict],
    transcripts: list[dict],
    news_articles: list[dict],
) -> list[dict]:
    """Parse all raw data into section-segmented documents."""
    all_documents = []

    # Parse SEC filings
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

    # Parse news
    parsed_news = parse_news_articles(news_articles)
    all_documents.extend(parsed_news)

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


def _save_financials(financials_by_ticker: dict):
    """Save structured financial data as JSON (not chunked)."""
    fin_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    for ticker, data in financials_by_ticker.items():
        path = fin_dir / ticker / "all_financials.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


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
    sec_filings, transcripts, news, financials = await _fetch_all_data(tickers)
    _save_financials(financials)
    logger.info(
        f"Fetched: {len(sec_filings)} filings, "
        f"{len(transcripts)} transcripts, {len(news)} news articles"
    )

    # Step 2: Parse
    logger.info("Step 2: Parsing documents...")
    documents = _parse_all_data(sec_filings, transcripts, news)

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
