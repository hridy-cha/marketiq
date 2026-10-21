"""
Fundamental factors from raw quarterly statements, point-in-time.

Inputs are the `FundamentalPayload` rows (income, balance, cashflow, optional ratios) filed on or
before the as-of date. Trailing-twelve-month (TTM) figures are the sum of the last four quarters.
Valuation multiples use the as-of close price and TTM per-share values. Any factor that cannot be
computed is None, and the caller treats the component as partially available.

Field names follow FMP conventions; the yfinance provider maps its statements to the same keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FACTOR_LABELS = {
    "revenue_growth_yoy": "Revenue growth (TTM YoY)",
    "eps_growth_yoy": "EPS growth (TTM YoY)",
    "gross_margin": "Gross margin (TTM)",
    "operating_margin": "Operating margin (TTM)",
    "net_margin": "Net margin (TTM)",
    "roe": "Return on equity (TTM)",
    "debt_to_equity": "Debt to equity",
    "current_ratio": "Current ratio",
    "fcf_per_share": "Free cash flow per share (TTM)",
    "fcf_yield": "FCF yield",
    "pe_ratio": "P/E (TTM)",
    "price_to_book": "Price to book",
    "price_to_sales": "Price to sales (TTM)",
    "peg": "PEG (P/E over EPS growth)",
    "dividend_yield": "Dividend yield (TTM)",
}


@dataclass
class FundamentalSnapshot:
    symbol: str
    as_of: date
    last_period_end: Optional[date]
    last_filing_date: Optional[date]
    quarters_available: int
    factors: Dict[str, Optional[float]] = field(default_factory=dict)
    ttm: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["last_period_end"] = self.last_period_end.isoformat() if self.last_period_end else None
        d["last_filing_date"] = self.last_filing_date.isoformat() if self.last_filing_date else None
        d["labels"] = FACTOR_LABELS
        return d


def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _frame(payloads, statement: str) -> pd.DataFrame:
    rows = [{"period_end": p.period_end, "filing_date": p.filing_date, **p.data} for p in payloads if p.statement == statement]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("period_end").drop_duplicates("period_end", keep="last")
    return df.reset_index(drop=True)


def _ttm(df: pd.DataFrame, col: str, offset: int = 0) -> Optional[float]:
    """Sum of 4 quarters ending `offset` quarters before the latest. None if fewer than 4 present."""
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if offset:
        s = s.iloc[:-offset] if offset < len(s) else s.iloc[:0]
    tail = s.tail(4)
    if len(tail) < 4 or tail.isna().any():
        return None
    return float(tail.sum())


def _latest(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if s.empty else float(s.iloc[-1])


def _growth(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev is None or prev == 0:
        return None
    if prev < 0:  # growth from a negative base is not meaningful as a ratio
        return None
    return (cur - prev) / abs(prev)


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_fundamentals(
    symbol: str, payloads: list, as_of: date, price: Optional[float], max_staleness_days: int = 400
) -> FundamentalSnapshot:
    income = _frame(payloads, "income")
    balance = _frame(payloads, "balance")
    cashflow = _frame(payloads, "cashflow")
    ratios = _frame(payloads, "ratios")

    notes: List[str] = []
    all_periods = [p.period_end for p in payloads]
    last_period = max(all_periods) if all_periods else None
    filings = [p.filing_date for p in payloads if p.filing_date]
    last_filing = max(filings) if filings else None
    snap = FundamentalSnapshot(symbol, as_of, last_period, last_filing, len(income), notes=notes)

    if income.empty and balance.empty:
        notes.append("No financial statements available")
        return snap
    if last_period and (as_of - last_period).days > max_staleness_days:
        notes.append(f"Latest statement period {last_period} is older than {max_staleness_days} days; fundamentals ignored")
        return snap

    rev_ttm = _ttm(income, "revenue")
    rev_prev = _ttm(income, "revenue", offset=4)
    ni_ttm = _ttm(income, "netIncome")
    eps_col = "epsdiluted" if "epsdiluted" in income.columns and income["epsdiluted"].notna().any() else "eps"
    eps_ttm = _ttm(income, eps_col)
    eps_prev = _ttm(income, eps_col, offset=4)
    gp_ttm = _ttm(income, "grossProfit")
    op_ttm = _ttm(income, "operatingIncome")
    shares = _latest(income, "weightedAverageShsOutDil") or _latest(balance, "commonStockSharesOutstanding")
    equity = _latest(balance, "totalStockholdersEquity")
    total_debt = _latest(balance, "totalDebt")
    if total_debt is None:
        lt, st = _latest(balance, "longTermDebt"), _latest(balance, "shortTermDebt")
        total_debt = (lt or 0.0) + (st or 0.0) if (lt is not None or st is not None) else None
    cur_assets = _latest(balance, "totalCurrentAssets")
    cur_liab = _latest(balance, "totalCurrentLiabilities")
    fcf_ttm = _ttm(cashflow, "freeCashFlow")
    if fcf_ttm is None:
        ocf, capex = _ttm(cashflow, "operatingCashFlow"), _ttm(cashflow, "capitalExpenditure")
        fcf_ttm = ocf + capex if (ocf is not None and capex is not None) else None  # capex is negative in FMP
    div_ttm = _ttm(cashflow, "dividendsPaid")

    f: Dict[str, Optional[float]] = {}
    f["revenue_growth_yoy"] = _growth(rev_ttm, rev_prev)
    f["eps_growth_yoy"] = _growth(eps_ttm, eps_prev)
    f["gross_margin"] = _ratio(gp_ttm, rev_ttm)
    f["operating_margin"] = _ratio(op_ttm, rev_ttm)
    f["net_margin"] = _ratio(ni_ttm, rev_ttm)
    f["roe"] = _ratio(ni_ttm, equity) if equity and equity > 0 else None
    f["debt_to_equity"] = _ratio(total_debt, equity) if equity and equity > 0 else None
    f["current_ratio"] = _ratio(cur_assets, cur_liab)
    f["fcf_per_share"] = _ratio(fcf_ttm, shares)
    mcap = price * shares if (price and shares) else None
    f["fcf_yield"] = _ratio(fcf_ttm, mcap)
    f["pe_ratio"] = _ratio(price, eps_ttm) if (price and eps_ttm and eps_ttm > 0) else None
    bvps = _ratio(equity, shares)
    f["price_to_book"] = _ratio(price, bvps) if (price and bvps and bvps > 0) else None
    f["price_to_sales"] = _ratio(mcap, rev_ttm)
    g = f["eps_growth_yoy"]
    f["peg"] = _ratio(f["pe_ratio"], g * 100.0) if (f["pe_ratio"] and g and g > 0) else None
    f["dividend_yield"] = _ratio(abs(div_ttm), mcap) if (div_ttm is not None and mcap) else None

    # Fall back to provider ratios for anything still missing (point-in-time rows only).
    if not ratios.empty:
        fallback = {
            "current_ratio": "currentRatio",
            "debt_to_equity": "debtEquityRatio",
            "roe": "returnOnEquity",
            "net_margin": "netProfitMargin",
            "gross_margin": "grossProfitMargin",
            "operating_margin": "operatingProfitMargin",
            "fcf_per_share": "freeCashFlowPerShare",
            "dividend_yield": "dividendYield",
        }
        for k, col in fallback.items():
            if f.get(k) is None:
                v = _latest(ratios, col)
                if v is not None:
                    f[k] = v
                    notes.append(f"{k} taken from provider ratios")

    snap.factors = {k: _num(v) for k, v in f.items()}
    snap.ttm = {
        "revenue": rev_ttm,
        "net_income": ni_ttm,
        "eps": eps_ttm,
        "free_cash_flow": fcf_ttm,
        "shares": shares,
        "equity": equity,
        "total_debt": total_debt,
        "market_cap": mcap,
    }
    if rev_prev is None:
        notes.append("Fewer than 8 quarters: YoY growth unavailable")
    return snap


def universe_percentiles(snapshots: Dict[str, FundamentalSnapshot], directions: Dict[str, int]) -> Dict[str, Dict[str, Optional[float]]]:
    """Percentile rank (0-100) of each factor across the universe, higher = better per `directions`.

    Returns {symbol: {factor: percentile}}; None when the factor is missing for that symbol."""
    rows = {sym: snap.factors for sym, snap in snapshots.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    out: Dict[str, Dict[str, Optional[float]]] = {sym: {} for sym in snapshots}
    for factor, direction in directions.items():
        if factor not in df.columns:
            for sym in snapshots:
                out[sym][factor] = None
            continue
        col = pd.to_numeric(df[factor], errors="coerce")
        valid = col.dropna()
        if len(valid) < 3:
            for sym in snapshots:
                out[sym][factor] = None
            continue
        ranks = valid.rank(pct=True, ascending=(direction > 0)) * 100.0
        for sym in snapshots:
            out[sym][factor] = float(ranks[sym]) if sym in ranks.index else None
    return out
