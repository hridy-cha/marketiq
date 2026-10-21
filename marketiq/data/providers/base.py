"""Provider interfaces. Every external data source implements one or more of these."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Protocol, runtime_checkable

import pandas as pd


@dataclass
class NewsItem:
    symbol: str
    published_at: datetime
    title: str
    summary: str = ""
    url: str = ""
    publisher: str = ""
    source: str = ""


@dataclass
class FundamentalsBundle:
    """Raw statement rows as returned by a provider, keyed by statement type."""

    symbol: str
    statements: dict = field(default_factory=dict)  # statement -> list[dict] each with 'date'
    earnings: List[dict] = field(default_factory=list)  # [{'event_date': date, 'eps_estimate': ...}]
    source: str = ""


@runtime_checkable
class PriceProvider(Protocol):
    name: str

    def fetch_prices(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return a frame with columns date, open, high, low, close, adj_close, volume."""
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    name: str

    def fetch_fundamentals(self, symbol: str) -> FundamentalsBundle: ...


@runtime_checkable
class NewsProvider(Protocol):
    name: str

    def fetch_news(self, symbol: str, start: date, end: date) -> List[NewsItem]: ...


class ProviderError(RuntimeError):
    pass


def empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, rename common variants, sort, drop rows without close."""
    if df is None or df.empty:
        return empty_price_frame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and "date" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "date"})
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    out = out.rename(columns={"adjclose": "adj_close", "adjusted_close": "adj_close", "datetime": "date"})
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    out = out.dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def provider_for(kind: str, name: Optional[str] = None):
    """Factory. kind in {price, fundamentals, news}. Falls back to yfinance."""
    from marketiq.config.settings import get_settings

    s = get_settings()
    name = (name or (s.data.price_provider if kind in ("price", "fundamentals") else s.data.news_provider)).lower()
    if name == "fmp":
        from marketiq.data.providers.fmp import FMPProvider

        return FMPProvider()
    if name == "yfinance":
        from marketiq.data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    raise ProviderError(f"Unknown provider '{name}' for {kind}")
