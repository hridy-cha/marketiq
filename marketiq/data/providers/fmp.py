"""
Financial Modeling Prep provider (API key required).

Endpoint usage adapted from FinRL-X `src/data/data_fetcher.py` (Apache 2.0, AI4Finance Foundation):
historical-price-eod/full, income-statement, balance-sheet-statement, cash-flow-statement, ratios,
earnings, news/stock. Requests are plain HTTP so the module is easy to stub in tests.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from marketiq.config.settings import get_settings
from marketiq.data.providers.base import (
    FundamentalsBundle,
    NewsItem,
    ProviderError,
    empty_price_frame,
    normalize_price_frame,
)

logger = logging.getLogger(__name__)

_STATEMENT_ENDPOINTS = {
    "income": "income-statement",
    "balance": "balance-sheet-statement",
    "cashflow": "cash-flow-statement",
    "ratios": "ratios",
}


class FMPProvider:
    name = "fmp"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, session=None):
        s = get_settings().fmp
        self.api_key = api_key or (s.api_key.get_secret_value() if s.api_key else None)
        if not self.api_key:
            raise ProviderError("FMP_API_KEY is not configured")
        self.base_url = (base_url or s.base_url).rstrip("/")
        self.http = session or requests.Session()

    # ---- helpers ---------------------------------------------------------
    def _get(self, endpoint: str, **params) -> Any:
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        resp = self.http.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ---- prices ----------------------------------------------------------
    def fetch_prices(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        data = self._get("historical-price-eod/full", symbol=symbol, **{"from": start.isoformat(), "to": end.isoformat()})
        rows = data.get("historical", data) if isinstance(data, dict) else data
        if not rows:
            return empty_price_frame()
        df = pd.DataFrame(rows)
        if "adjClose" in df.columns:
            df = df.rename(columns={"adjClose": "adj_close"})
        return normalize_price_frame(df)

    # ---- fundamentals ----------------------------------------------------
    def fetch_fundamentals(self, symbol: str, limit: int = 40) -> FundamentalsBundle:
        bundle = FundamentalsBundle(symbol=symbol, source=self.name)
        for key, endpoint in _STATEMENT_ENDPOINTS.items():
            try:
                rows = self._get(endpoint, symbol=symbol, period="quarter", limit=limit) or []
                bundle.statements[key] = rows if isinstance(rows, list) else []
            except requests.RequestException as exc:
                logger.warning("FMP %s failed for %s: %s", endpoint, symbol, exc)
        try:
            earn = self._get("earnings", symbol=symbol, limit=8) or []
            for e in earn:
                if e.get("date"):
                    bundle.earnings.append(
                        {
                            "event_date": pd.Timestamp(e["date"]).date(),
                            "eps_estimate": _f(e.get("epsEstimated")),
                            "eps_actual": _f(e.get("epsActual")),
                            "revenue_estimate": _f(e.get("revenueEstimated")),
                            "revenue_actual": _f(e.get("revenueActual")),
                        }
                    )
        except requests.RequestException as exc:
            logger.warning("FMP earnings failed for %s: %s", symbol, exc)
        return bundle

    # ---- news ------------------------------------------------------------
    def fetch_news(self, symbol: str, start: date, end: date) -> List[NewsItem]:
        try:
            payload = self._get("news/stock", symbols=symbol, **{"from": start.isoformat(), "to": end.isoformat()})
        except requests.RequestException as exc:
            logger.warning("FMP news failed for %s: %s", symbol, exc)
            return []
        rows = payload.get("news", payload) if isinstance(payload, dict) else payload
        items: List[NewsItem] = []
        for n in rows or []:
            if not n.get("publishedDate"):
                continue
            items.append(
                NewsItem(
                    symbol=symbol,
                    published_at=pd.Timestamp(n["publishedDate"]).to_pydatetime(),
                    title=n.get("title", "") or "",
                    summary=n.get("text", "") or "",
                    url=n.get("url", "") or "",
                    publisher=n.get("site", "") or n.get("publisher", "") or "",
                    source=self.name,
                )
            )
        return items


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
