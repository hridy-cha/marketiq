"""Repository: typed read/write helpers over the ORM. Application code uses these, never raw SQL."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from marketiq.data.models import (
    EarningsEvent,
    FundamentalPayload,
    NewsArticle,
    NewsFetchLog,
    PriceBar,
    ScoreSnapshot,
    SignalSnapshot,
    Ticker,
    WatchlistItem,
)

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
def upsert_tickers(session: Session, rows: Iterable[dict]) -> int:
    n = 0
    for r in rows:
        obj = session.get(Ticker, r["symbol"])
        if obj is None:
            obj = Ticker(symbol=r["symbol"])
            session.add(obj)
        obj.name = r.get("name", obj.name or "")
        obj.sector = r.get("sector", obj.sector or "")
        obj.is_benchmark = bool(r.get("is_benchmark", False))
        n += 1
    session.flush()
    return n


def list_tickers(session: Session) -> List[Ticker]:
    return list(session.scalars(select(Ticker).order_by(Ticker.symbol)))


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def upsert_price_bars(session: Session, df: pd.DataFrame, symbol: str, source: str = "") -> int:
    """Insert-or-replace bars. `df` must have columns date, open, high, low, close, adj_close, volume."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "symbol": symbol,
                "date": row.date,
                "open": _f(row.open),
                "high": _f(row.high),
                "low": _f(row.low),
                "close": _f(row.close),
                "adj_close": _f(getattr(row, "adj_close", None)),
                "volume": _f(row.volume),
                "source": source,
            }
        )
    bind = session.get_bind()
    if bind.dialect.name == "sqlite":
        stmt = sqlite.insert(PriceBar).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "date"],
            set_={c: getattr(stmt.excluded, c) for c in ["open", "high", "low", "close", "adj_close", "volume", "source"]},
        )
        session.execute(stmt)
    else:  # pragma: no cover - exercised only on Postgres
        from sqlalchemy.dialects import postgresql

        stmt = postgresql.insert(PriceBar).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "date"],
            set_={c: getattr(stmt.excluded, c) for c in ["open", "high", "low", "close", "adj_close", "volume", "source"]},
        )
        session.execute(stmt)
    session.flush()
    return len(records)


def get_price_frame(
    session: Session, symbol: str, start: Optional[date] = None, end: Optional[date] = None
) -> pd.DataFrame:
    """OHLCV frame indexed by date (ascending). Empty frame with the right columns if nothing stored."""
    stmt = select(PriceBar).where(PriceBar.symbol == symbol)
    if start is not None:
        stmt = stmt.where(PriceBar.date >= start)
    if end is not None:
        stmt = stmt.where(PriceBar.date <= end)
    stmt = stmt.order_by(PriceBar.date)
    rows = session.scalars(stmt).all()
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS[1:], index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "adj_close": r.adj_close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def get_price_dates(session: Session, symbol: str) -> List[date]:
    return list(session.scalars(select(PriceBar.date).where(PriceBar.symbol == symbol).order_by(PriceBar.date)))


def price_coverage(session: Session) -> pd.DataFrame:
    stmt = select(
        PriceBar.symbol, func.count(PriceBar.id), func.min(PriceBar.date), func.max(PriceBar.date)
    ).group_by(PriceBar.symbol)
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["symbol", "bars", "first", "last"])


def latest_price_date(session: Session, symbol: str) -> Optional[date]:
    return session.scalar(select(func.max(PriceBar.date)).where(PriceBar.symbol == symbol))


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def upsert_fundamental_payloads(session: Session, rows: Iterable[dict]) -> int:
    n = 0
    for r in rows:
        existing = session.scalar(
            select(FundamentalPayload).where(
                FundamentalPayload.symbol == r["symbol"],
                FundamentalPayload.statement == r["statement"],
                FundamentalPayload.period_end == r["period_end"],
            )
        )
        if existing is None:
            session.add(FundamentalPayload(**r))
        else:
            existing.data = r["data"]
            existing.filing_date = r.get("filing_date")
            existing.source = r.get("source", existing.source)
        n += 1
    session.flush()
    return n


def get_fundamental_payloads(
    session: Session, symbol: str, statement: Optional[str] = None, as_of: Optional[date] = None
) -> List[FundamentalPayload]:
    """Payloads ordered by period_end ascending. With `as_of`, only rows whose filing date (or period
    end when the filing date is unknown) is on or before `as_of`, which keeps point-in-time discipline."""
    stmt = select(FundamentalPayload).where(FundamentalPayload.symbol == symbol)
    if statement:
        stmt = stmt.where(FundamentalPayload.statement == statement)
    rows = list(session.scalars(stmt.order_by(FundamentalPayload.period_end)))
    if as_of is not None:
        rows = [r for r in rows if (r.filing_date or r.period_end) <= as_of]
    return rows


def upsert_earnings(session: Session, rows: Iterable[dict]) -> int:
    n = 0
    for r in rows:
        existing = session.scalar(
            select(EarningsEvent).where(EarningsEvent.symbol == r["symbol"], EarningsEvent.event_date == r["event_date"])
        )
        if existing is None:
            session.add(EarningsEvent(**r))
        else:
            for k, v in r.items():
                setattr(existing, k, v)
        n += 1
    session.flush()
    return n


def next_earnings(session: Session, symbol: str, as_of: date) -> Optional[EarningsEvent]:
    return session.scalar(
        select(EarningsEvent)
        .where(EarningsEvent.symbol == symbol, EarningsEvent.event_date >= as_of)
        .order_by(EarningsEvent.event_date)
        .limit(1)
    )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def upsert_news(session: Session, rows: Iterable[dict]) -> int:
    n = 0
    for r in rows:
        existing = session.scalar(
            select(NewsArticle).where(
                NewsArticle.symbol == r["symbol"],
                NewsArticle.published_at == r["published_at"],
                NewsArticle.title == r.get("title", ""),
            )
        )
        if existing is None:
            session.add(NewsArticle(**r))
            n += 1
        else:
            for k in ("summary", "url", "publisher", "sentiment_label", "sentiment_score", "sentiment_model"):
                if r.get(k) is not None:
                    setattr(existing, k, r[k])
    session.flush()
    return n


def get_news(session: Session, symbol: str, start: datetime, end: datetime) -> List[NewsArticle]:
    stmt = (
        select(NewsArticle)
        .where(NewsArticle.symbol == symbol, NewsArticle.published_at >= start, NewsArticle.published_at <= end)
        .order_by(NewsArticle.published_at.desc())
    )
    return list(session.scalars(stmt))


def log_news_fetch(session: Session, symbol: str, start: date, end: date, count: int) -> None:
    session.add(NewsFetchLog(symbol=symbol, start_date=start, end_date=end, article_count=count))
    session.flush()


def news_recently_fetched(session: Session, symbol: str, within_hours: int = 12) -> bool:
    cutoff = datetime.utcnow() - timedelta(hours=within_hours)
    return session.scalar(
        select(func.count(NewsFetchLog.id)).where(NewsFetchLog.symbol == symbol, NewsFetchLog.fetched_at >= cutoff)
    ) > 0


# ---------------------------------------------------------------------------
# Scores and signals
# ---------------------------------------------------------------------------
def save_score(session: Session, **fields) -> ScoreSnapshot:
    existing = session.scalar(
        select(ScoreSnapshot).where(ScoreSnapshot.symbol == fields["symbol"], ScoreSnapshot.as_of == fields["as_of"])
    )
    if existing is None:
        existing = ScoreSnapshot(**fields)
        session.add(existing)
    else:
        for k, v in fields.items():
            setattr(existing, k, v)
    session.flush()
    return existing


def save_signal(session: Session, **fields) -> SignalSnapshot:
    existing = session.scalar(
        select(SignalSnapshot).where(SignalSnapshot.symbol == fields["symbol"], SignalSnapshot.as_of == fields["as_of"])
    )
    if existing is None:
        existing = SignalSnapshot(**fields)
        session.add(existing)
    else:
        for k, v in fields.items():
            setattr(existing, k, v)
    session.flush()
    return existing


def latest_scores(session: Session, as_of: Optional[date] = None) -> List[ScoreSnapshot]:
    """Latest score per symbol (or the snapshot for `as_of`)."""
    if as_of is None:
        as_of = session.scalar(select(func.max(ScoreSnapshot.as_of)))
        if as_of is None:
            return []
    return list(session.scalars(select(ScoreSnapshot).where(ScoreSnapshot.as_of == as_of).order_by(ScoreSnapshot.score.desc())))


def latest_signals(session: Session, as_of: Optional[date] = None) -> Dict[str, SignalSnapshot]:
    if as_of is None:
        as_of = session.scalar(select(func.max(SignalSnapshot.as_of)))
        if as_of is None:
            return {}
    return {s.symbol: s for s in session.scalars(select(SignalSnapshot).where(SignalSnapshot.as_of == as_of))}


def previous_score(session: Session, symbol: str, before: date) -> Optional[ScoreSnapshot]:
    return session.scalar(
        select(ScoreSnapshot)
        .where(ScoreSnapshot.symbol == symbol, ScoreSnapshot.as_of < before)
        .order_by(ScoreSnapshot.as_of.desc())
        .limit(1)
    )


def score_history(session: Session, symbol: str, limit: int = 60) -> List[ScoreSnapshot]:
    rows = session.scalars(
        select(ScoreSnapshot).where(ScoreSnapshot.symbol == symbol).order_by(ScoreSnapshot.as_of.desc()).limit(limit)
    ).all()
    return list(reversed(rows))


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
def list_watchlist(session: Session) -> List[WatchlistItem]:
    return list(session.scalars(select(WatchlistItem).order_by(WatchlistItem.created_at)))


def add_watchlist(session: Session, symbol: str, note: str = "", added_by: str = "user") -> WatchlistItem:
    existing = session.scalar(select(WatchlistItem).where(WatchlistItem.symbol == symbol))
    if existing:
        if note:
            existing.note = note
        return existing
    item = WatchlistItem(symbol=symbol, note=note, added_by=added_by)
    session.add(item)
    session.flush()
    return item


def remove_watchlist(session: Session, symbol: str) -> bool:
    res = session.execute(delete(WatchlistItem).where(WatchlistItem.symbol == symbol))
    return res.rowcount > 0


def _f(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
