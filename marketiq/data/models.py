"""
SQLAlchemy models.

Schema concepts (price bars keyed by ticker+date, raw fundamentals payloads, news articles with a
fetch log) are carried over from FinRL-X `src/data/data_store.py` (Apache 2.0) and re-expressed as
ORM models so the same code runs on SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.utcnow()


class Ticker(Base):
    __tablename__ = "tickers"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    sector: Mapped[str] = mapped_column(String(64), default="")
    is_benchmark: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PriceBar(Base):
    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("symbol", "date", name="uq_price_symbol_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="")


class FundamentalPayload(Base):
    """One raw statement row (income, balance, cashflow, ratios, profile) per symbol per period."""

    __tablename__ = "fundamental_payloads"
    __table_args__ = (UniqueConstraint("symbol", "statement", "period_end", name="uq_fund_payload"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    statement: Mapped[str] = mapped_column(String(16), nullable=False)  # income|balance|cashflow|ratios|profile
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    filing_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), default="")
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class EarningsEvent(Base):
    __tablename__ = "earnings_events"
    __table_args__ = (UniqueConstraint("symbol", "event_date", name="uq_earnings_symbol_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    eps_estimate: Mapped[float | None] = mapped_column(Float)
    eps_actual: Mapped[float | None] = mapped_column(Float)
    revenue_estimate: Mapped[float | None] = mapped_column(Float)
    revenue_actual: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="")


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (UniqueConstraint("symbol", "published_at", "title", name="uq_news"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    publisher: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="")
    sentiment_label: Mapped[str | None] = mapped_column(String(16))
    sentiment_score: Mapped[float | None] = mapped_column(Float)  # -1 .. 1
    sentiment_model: Mapped[str | None] = mapped_column(String(64))


class NewsFetchLog(Base):
    __tablename__ = "news_fetch_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    article_count: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ScoreSnapshot(Base):
    """Composite score with component breakdown and explanation, as of a trading date."""

    __tablename__ = "score_snapshots"
    __table_args__ = (UniqueConstraint("symbol", "as_of", name="uq_score_symbol_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    as_of: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    components: Mapped[dict] = mapped_column(JSON, nullable=False)
    explanation: Mapped[list] = mapped_column(JSON, nullable=False)
    indicators: Mapped[dict] = mapped_column(JSON, default=dict)
    patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SignalSnapshot(Base):
    __tablename__ = "signal_snapshots"
    __table_args__ = (UniqueConstraint("symbol", "as_of", name="uq_signal_symbol_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    as_of: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)  # BUY|SELL|HOLD|WATCH
    rule: Mapped[str] = mapped_column(String(64), default="")
    score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON, nullable=False)
    risk_flags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class WatchlistItem(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="")
    added_by: Mapped[str] = mapped_column(String(16), default="user")  # user | system
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class MLModelRecord(Base):
    __tablename__ = "ml_models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(64))
    horizon_days: Mapped[int] = mapped_column(Integer)
    threshold: Mapped[float] = mapped_column(Float)
    train_start: Mapped[date | None] = mapped_column(Date)
    train_end: Mapped[date | None] = mapped_column(Date)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    features: Mapped[list] = mapped_column(JSON, default=list)
    path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    __table_args__ = (UniqueConstraint("model_version", "symbol", "as_of", name="uq_pred"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[date] = mapped_column(Date, index=True)
    prob_up: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[int] = mapped_column(Integer)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32), default="signal")  # signal | walk_forward
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    equity_curve: Mapped[list] = mapped_column(JSON, default=list)
    trades: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PaperAccount(Base):
    __tablename__ = "paper_accounts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    starting_cash: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    positions: Mapped[list["PaperPosition"]] = relationship(back_populates="account", cascade="all, delete-orphan")
    trades: Mapped[list["PaperTrade"]] = relationship(back_populates="account", cascade="all, delete-orphan")


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol", name="uq_paper_pos"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    account: Mapped[PaperAccount] = relationship(back_populates="positions")


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    trade_date: Mapped[date] = mapped_column(Date)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    account: Mapped[PaperAccount] = relationship(back_populates="trades")


class PaperEquityPoint(Base):
    __tablename__ = "paper_equity"
    __table_args__ = (UniqueConstraint("account_id", "as_of", name="uq_paper_equity"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    as_of: Mapped[date] = mapped_column(Date)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)


Index("ix_price_symbol_date", PriceBar.symbol, PriceBar.date)
Index("ix_news_symbol_published", NewsArticle.symbol, NewsArticle.published_at)
