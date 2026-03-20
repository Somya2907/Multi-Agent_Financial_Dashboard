"""Compute key financial metrics from cached structured financial data."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FINANCIALS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "financials"


def _safe_pct(a, b) -> float | None:
    """Compute (a - b) / |b| as a percentage, or None if not computable."""
    try:
        if b and b != 0:
            return round((a - b) / abs(b) * 100, 2)
    except (TypeError, ZeroDivisionError):
        pass
    return None


def _latest(records: list[dict], field: str):
    """Get the most recent non-None value of a field from a list of period records."""
    for rec in records:
        val = rec.get(field)
        if val is not None:
            return val
    return None


def compute_metrics(ticker: str) -> dict:
    """Compute key financial metrics from cached FMP data for a company.

    Reads from data/raw/financials/{ticker}/all_financials.json.
    Returns a dict of computed metrics; missing values are None.
    """
    path = FINANCIALS_DIR / ticker / "all_financials.json"
    if not path.exists():
        logger.warning(f"No financials cached for {ticker}")
        return {"error": f"No financial data available for {ticker}"}

    with open(path) as f:
        data = json.load(f)

    income = data.get("income_statement", [])
    balance = data.get("balance_sheet", [])
    cash_flow = data.get("cash_flow", [])
    ratios = data.get("ratios", [])
    key_metrics = data.get("key_metrics", [])

    # ---- Revenue Growth ----
    revenue_growth = None
    if len(income) >= 2:
        r0 = income[0].get("revenue") or income[0].get("totalRevenue")
        r1 = income[1].get("revenue") or income[1].get("totalRevenue")
        revenue_growth = _safe_pct(r0, r1)

    # ---- Profitability ----
    gross_margin = _latest(ratios, "grossProfitMargin")
    operating_margin = _latest(ratios, "operatingProfitMargin")
    net_margin = _latest(ratios, "netProfitMargin")

    # ---- Leverage ----
    debt_to_equity = _latest(ratios, "debtEquityRatio")
    interest_coverage = _latest(ratios, "interestCoverageRatio")

    total_debt = _latest(balance, "totalDebt") or _latest(balance, "longTermDebt")
    total_equity = (
        _latest(balance, "totalStockholdersEquity")
        or _latest(balance, "stockholdersEquity")
    )

    # ---- Liquidity ----
    current_ratio = _latest(ratios, "currentRatio")
    quick_ratio = _latest(ratios, "quickRatio")

    # ---- Cash Flow ----
    operating_cash_flow = _latest(cash_flow, "operatingCashFlow")
    free_cash_flow = _latest(key_metrics, "freeCashFlowPerShare")
    capex = _latest(cash_flow, "capitalExpenditure")

    # ---- Valuation ----
    pe_ratio = _latest(ratios, "priceEarningsRatio")
    market_cap = _latest(key_metrics, "marketCap")

    # ---- Period coverage ----
    periods = [rec.get("date") or rec.get("calendarYear") for rec in income[:3] if rec]

    metrics = {
        "ticker": ticker,
        "periods_covered": [p for p in periods if p],
        "revenue_growth_pct_yoy": revenue_growth,
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": operating_margin,
        "net_margin_pct": net_margin,
        "debt_to_equity_ratio": debt_to_equity,
        "interest_coverage_ratio": interest_coverage,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow_per_share": free_cash_flow,
        "capex": capex,
        "pe_ratio": pe_ratio,
        "market_cap": market_cap,
    }

    # Strip Nones for cleaner output
    return {k: v for k, v in metrics.items() if v is not None}
