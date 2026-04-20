"""FastAPI backend for the Multi-Agent Financial RAG system.

Exposes the LangGraph pipeline over HTTP so the Next.js dashboard can call it.

Endpoints:
  POST /analyze_company — auto-generate full analysis for a ticker (cached)
  POST /followup        — ask a grounded follow-up against a prior analysis
  POST /query           — legacy: run the full multi-agent pipeline on a query
  GET  /companies       — list all configured tickers + names
  GET  /health          — liveness check

Start:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload --workers 1

Note: FAISS is not fork-safe — always use --workers 1.
"""

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.cache import get_cached, save_cached, invalidate
from config.companies import COMPANIES
from pipeline.agents.graph import run_query
from pipeline.reasoning.followup import run_followup

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Risk Analyst API",
    description="Multi-agent financial RAG system (CMU 11-766)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_KNOWN_TICKERS = {c["ticker"] for c in COMPANIES}

_DEFAULT_ANALYSIS_QUERY = (
    "Provide a comprehensive financial risk analysis of this company over the "
    "last 2 years. Cover liquidity, credit, market, regulatory, and "
    "macroeconomic risks with specific numbers and citations from recent "
    "10-K/10-Q filings and earnings calls."
)


# ── Request schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    ticker: str | None = None


class AnalyzeCompanyRequest(BaseModel):
    ticker: str
    refresh: bool = False  # Force cache bypass and regenerate


class FollowupRequest(BaseModel):
    query: str
    prior_state: dict
    max_tokens: int | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/analyze_company")
def analyze_company(req: AnalyzeCompanyRequest) -> dict:
    """Auto-generate a full financial analysis for one ticker.

    Checks the file cache first (24h TTL). On miss, runs the full LangGraph
    pipeline with a default analysis query and a [current_year-2, current_year]
    time filter, then stores the result.
    """
    ticker = req.ticker.strip().upper()
    if ticker not in _KNOWN_TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown ticker: {ticker}")

    if req.refresh:
        invalidate(ticker)
    else:
        cached = get_cached(ticker)
        if cached is not None:
            cached["_cache_hit"] = True
            return cached

    current_year = datetime.utcnow().year
    year_filter = (current_year - 2, current_year)

    logger.info(f"[API] /analyze_company {ticker} year_filter={year_filter} (cache miss)")
    try:
        result = run_query(
            query=_DEFAULT_ANALYSIS_QUERY,
            ticker=ticker,
            year_filter=year_filter,
        )
    except Exception as e:
        logger.exception(f"[API] /analyze_company {ticker} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    save_cached(ticker, result)
    result["_cache_hit"] = False
    return result


@app.post("/followup")
def followup(req: FollowupRequest) -> dict:
    """Answer a follow-up question against a prior /analyze_company result.

    Skips the multi-agent pipeline entirely — makes a single LLM call with the
    prior state's metrics, risk synthesis, qualitative analysis, and top
    retrieved chunks as context. Latency: ~3-8 seconds.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not isinstance(req.prior_state, dict) or not req.prior_state:
        raise HTTPException(status_code=400, detail="prior_state must be a non-empty object")

    logger.info(f"[API] /followup q={req.query!r} max_tokens={req.max_tokens}")
    try:
        kwargs = {"max_tokens": req.max_tokens} if req.max_tokens else {}
        return run_followup(req.query, req.prior_state, **kwargs)
    except Exception as e:
        logger.exception(f"[API] /followup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_endpoint(req: QueryRequest) -> dict:
    """Legacy endpoint: run the full multi-agent pipeline on a free-form query."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    logger.info(f"[API] /query q={req.query!r} ticker={req.ticker}")
    try:
        return run_query(query=req.query, ticker=req.ticker)
    except Exception as e:
        logger.exception(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
def list_companies() -> list[dict]:
    """Return all configured companies for the company selector dropdown."""
    return [
        {"ticker": c["ticker"], "name": c["name"], "sector": c["sector"]}
        for c in COMPANIES
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
