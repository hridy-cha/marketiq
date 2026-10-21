"""
Yahoo Finance provider (free, no key). Default live source.

Covers daily OHLCV, quarterly statements, the next earnings date and recent news. Statement rows
are re-keyed to the FMP-style field names MarketIQ's fundamentals module reads, so downstream code
does not care which provider produced them.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List

import pandas as pd

from marketiq.data.providers.base import (
    FundamentalsBundle,
    NewsItem,
    ProviderError,
    empty_price_frame,
    normalize_price_frame,
)

logger = logging.getLogger(__name__)

# yfinance statement index label -> MarketIQ/FMP-style key
_INCOME_MAP = {
    "Total Revenue": "revenue",
    "Gross Profit": "grossProfit",
    "Operating Income": "operatingIncome",
    "Net Income": "netIncome",
    "Diluted EPS": "epsdiluted",
    "Basic EPS": "eps",
    "EBITDA": "ebitda",
    "Diluted Average Shares": "weightedAverageShsOutDil",
}
_BALANCE_MAP = {
    "Total Assets": "totalAssets",
    "Total Liabilities Net Minority Interest": "totalLiabilities",
    "Stockholders Equity": "totalStockholdersEquity",
    "Current Assets": "totalCurrentAssets",
    "Current Liabilities": "totalCurrentLiabilities",
    "Total Debt": "totalDebt",
    "Cash And Cash Equivalents": "cashAndCashEquivalents",
    "Ordinary Shares Number": "commonStockSharesOutstanding",
}
_CASHFLOW_MAP = {
    "Operating Cash Flow": "operatingCashFlow",
    "Free Cash Flow": "freeCashFlow",
    "Capital Expenditure": "capitalExpenditure",
    "Cash Dividends Paid": "dividendsPaid",
}


def _statement_rows(df: pd.DataFrame, mapping: Dict[str, str]) -> List[dict]:
    rows: List[dict] = []
    if df is None or df.empty:
        return rows
    for col in df.columns:
        period_end = pd.Timestamp(col).date()
        row = {"date": period_end.isoformat()}
        for label, key in mapping.items():
            if label in df.index:
                val = df.at[label, col]
                row[key] = None if pd.isna(val) else float(val)
        rows.append(row)
    return rows


class YFinanceProvider:
    name = "yfinance"

    def __init__(self):
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("yfinance is not installed") from exc

    def fetch_prices(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        # yfinance's end is exclusive.
        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return empty_price_frame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        return normalize_price_frame(df)

    def fetch_fundamentals(self, symbol: str) -> FundamentalsBundle:
        import yfinance as yf

        t = yf.Ticker(symbol)
        bundle = FundamentalsBundle(symbol=symbol, source=self.name)
        try:
            bundle.statements["income"] = _statement_rows(t.quarterly_income_stmt, _INCOME_MAP)
            bundle.statements["balance"] = _statement_rows(t.quarterly_balance_sheet, _BALANCE_MAP)
            bundle.statements["cashflow"] = _statement_rows(t.quarterly_cashflow, _CASHFLOW_MAP)
        except Exception as exc:  # network / parsing
            logger.warning("yfinance statements failed for %s: %s", symbol, exc)
        try:
            cal = t.calendar or {}
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            for d in dates:
                if d:
                    bundle.earnings.append(
                        {
                            "event_date": pd.Timestamp(d).date(),
                            "eps_estimate": _f(cal.get("Earnings Average")),
                            "revenue_estimate": _f(cal.get("Revenue Average")),
                        }
                    )
        except Exception as exc:
            logger.warning("yfinance calendar failed for %s: %s", symbol, exc)
        return bundle

    def fetch_news(self, symbol: str, start: date, end: date) -> List[NewsItem]:
        import yfinance as yf

        items: List[NewsItem] = []
        try:
            raw = yf.Ticker(symbol).news or []
        except Exception as exc:
            logger.warning("yfinance news failed for %s: %s", symbol, exc)
            return items
        for n in raw:
            content = n.get("content", n)  # new-style payloads nest under 'content'
            ts = content.get("pubDate") or n.get("providerPublishTime")
            if ts is None:
                continue
            published = pd.Timestamp(ts, unit="s") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            published = published.tz_localize(None) if published.tzinfo else published
            if not (start <= published.date() <= end):
                continue
            provider = content.get("provider") or {}
            url = (content.get("canonicalUrl") or {}).get("url") or n.get("link", "")
            items.append(
                NewsItem(
                    symbol=symbol,
                    published_at=published.to_pydatetime(),
                    title=content.get("title") or n.get("title", ""),
                    summary=content.get("summary", "") or "",
                    url=url or "",
                    publisher=provider.get("displayName") if isinstance(provider, dict) else str(n.get("publisher", "")),
                    source=self.name,
                )
            )
        return items


def _f(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None
