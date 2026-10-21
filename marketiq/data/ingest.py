"""
Ingestion service: fills gaps in stored data from a live provider.

Prices: compares stored trading days against the exchange calendar and fetches only the missing
ranges (pattern from FinRL-X's incremental fetcher). Fundamentals and news: refreshed per symbol.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from marketiq.config.settings import get_settings
from marketiq.data import repository as repo
from marketiq.data.calendar import consolidate_ranges, last_completed_trading_day, missing_trading_days
from marketiq.data.providers.base import FundamentalsProvider, NewsProvider, PriceProvider

logger = logging.getLogger(__name__)


def fetch_missing_prices(
    session: Session,
    provider: PriceProvider,
    symbols: Iterable[str],
    start: date,
    end: Optional[date] = None,
    max_ranges_per_symbol: int = 3,
) -> Dict[str, int]:
    """Fetch and store bars for trading days not yet present. Returns bars stored per symbol."""
    exchange = get_settings().data.exchange
    end = end or last_completed_trading_day(exchange)
    stored: Dict[str, int] = {}
    for sym in symbols:
        have = repo.get_price_dates(session, sym)
        missing = missing_trading_days(have, start, end, exchange)
        if not missing:
            stored[sym] = 0
            continue
        ranges = consolidate_ranges(missing)
        # Many small gaps usually mean a provider calendar mismatch; fetch one span instead.
        if len(ranges) > max_ranges_per_symbol:
            ranges = [(ranges[0][0], ranges[-1][1])]
        total = 0
        for s, e in ranges:
            try:
                df = provider.fetch_prices(sym, pd.Timestamp(s).date(), pd.Timestamp(e).date())
            except Exception as exc:
                logger.warning("%s price fetch failed for %s %s..%s: %s", provider.name, sym, s, e, exc)
                continue
            total += repo.upsert_price_bars(session, df, sym, source=provider.name)
        stored[sym] = total
        logger.info("%s: stored %d bars from %s (%d missing days)", sym, total, provider.name, len(missing))
    return stored


def refresh_fundamentals(session: Session, provider: FundamentalsProvider, symbols: Iterable[str]) -> Dict[str, int]:
    stored: Dict[str, int] = {}
    for sym in symbols:
        try:
            bundle = provider.fetch_fundamentals(sym)
        except Exception as exc:
            logger.warning("%s fundamentals failed for %s: %s", provider.name, sym, exc)
            stored[sym] = 0
            continue
        records: List[dict] = []
        for statement, rows in bundle.statements.items():
            for row in rows:
                period_end = pd.to_datetime(row.get("date"), errors="coerce")
                if pd.isna(period_end):
                    continue
                filing = pd.to_datetime(row.get("fillingDate") or row.get("filingDate"), errors="coerce")
                records.append(
                    {
                        "symbol": sym,
                        "statement": statement,
                        "period_end": period_end.date(),
                        "filing_date": None if pd.isna(filing) else filing.date(),
                        "source": bundle.source,
                        "data": row,
                    }
                )
        n = repo.upsert_fundamental_payloads(session, records)
        if bundle.earnings:
            repo.upsert_earnings(session, [{"symbol": sym, "source": bundle.source, **e} for e in bundle.earnings])
        stored[sym] = n
    return stored


def refresh_news(
    session: Session,
    provider: NewsProvider,
    symbols: Iterable[str],
    lookback_days: int = 14,
    analyzer=None,
    force: bool = False,
) -> Dict[str, int]:
    """Fetch recent news; optionally score sentiment with `analyzer` (see analysis.sentiment)."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    stored: Dict[str, int] = {}
    for sym in symbols:
        if not force and repo.news_recently_fetched(session, sym):
            stored[sym] = 0
            continue
        try:
            items = provider.fetch_news(sym, start, end)
        except Exception as exc:
            logger.warning("%s news failed for %s: %s", provider.name, sym, exc)
            stored[sym] = 0
            continue
        rows = []
        for it in items:
            row = {
                "symbol": it.symbol,
                "published_at": it.published_at,
                "publisher": it.publisher,
                "title": it.title,
                "summary": it.summary,
                "url": it.url,
                "source": it.source,
            }
            if analyzer is not None:
                res = analyzer.analyze(f"{it.title}. {it.summary}")
                row.update(sentiment_label=res.label, sentiment_score=res.score, sentiment_model=res.model)
            rows.append(row)
        n = repo.upsert_news(session, rows)
        repo.log_news_fetch(session, sym, start, end, len(items))
        stored[sym] = n
    return stored
