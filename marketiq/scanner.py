"""
Scan orchestrator.

`ScanContext` loads everything once (prices, fundamentals payloads, news, optional ML predictions)
and can evaluate the universe as of any trading date using only data available at that date. The
CLI `scan` command evaluates the latest date and persists score and signal snapshots; the backtester
calls `evaluate` for many dates with the same context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from marketiq.data.models import EarningsEvent

from marketiq.analysis.fundamentals import FundamentalSnapshot, build_statement_frames, compute_fundamentals_from_frames, universe_percentiles
from marketiq.analysis.indicators import compute_indicators
from marketiq.analysis.patterns import detect_all
from marketiq.analysis.sentiment import AggregateSentiment, aggregate_sentiment
from marketiq.config.loaders import ScoringConfig, SignalConfig, Universe, load_scoring_config, load_signal_config, load_universe
from marketiq.data import repository as repo
from marketiq.scoring.scorer import ScoreResult, score_symbol
from marketiq.signals.engine import SignalResult, evaluate as evaluate_signal

logger = logging.getLogger(__name__)

MLPredictor = Callable[[str, pd.DataFrame, Optional[pd.DataFrame]], Optional[float]]  # (symbol, ind, bench_ind) -> prob_up


@dataclass
class ScanRow:
    symbol: str
    as_of: str
    score: ScoreResult
    signal: SignalResult
    indicators: dict
    patterns: dict
    fundamentals: Optional[FundamentalSnapshot]
    sentiment: Optional[AggregateSentiment]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "score": self.score.to_dict(),
            "signal": self.signal.to_dict(),
            "indicators": self.indicators,
            "patterns": self.patterns,
            "fundamentals": self.fundamentals.to_dict() if self.fundamentals else None,
            "sentiment": self.sentiment.__dict__ if self.sentiment else None,
        }


@dataclass
class ScanContext:
    universe: Universe
    scoring: ScoringConfig
    signals: SignalConfig
    prices: Dict[str, pd.DataFrame] = field(default_factory=dict)
    indicators: Dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)  # symbol -> statement frames
    news: Dict[str, list] = field(default_factory=dict)
    earnings: Dict[str, List[date]] = field(default_factory=dict)
    ml_predictor: Optional[MLPredictor] = None
    ml_horizon: Optional[int] = None
    ml_status: str = ""

    @classmethod
    def load(
        cls,
        session: Session,
        universe: Optional[Universe] = None,
        scoring: Optional[ScoringConfig] = None,
        signals: Optional[SignalConfig] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        with_news: bool = True,
        ml_predictor: Optional[MLPredictor] = None,
        ml_horizon: Optional[int] = None,
        ml_status: str = "",
    ) -> "ScanContext":
        universe = universe or load_universe()
        ctx = cls(universe, scoring or load_scoring_config(), signals or load_signal_config(),
                  ml_predictor=ml_predictor, ml_horizon=ml_horizon, ml_status=ml_status)
        for sym in universe.all_symbols:
            df = repo.get_price_frame(session, sym, start=start, end=end)
            if df.empty:
                logger.warning("No prices for %s; skipping", sym)
                continue
            ctx.prices[sym] = df
            ctx.indicators[sym] = compute_indicators(df)
        for sym in universe.symbols:
            ctx.fundamentals[sym] = build_statement_frames(repo.get_fundamental_payloads(session, sym),
                                                           lag_days=repo.UNKNOWN_FILING_LAG_DAYS)
            events = session.scalars(
                select(EarningsEvent.event_date).where(EarningsEvent.symbol == sym)
            ).all()
            ctx.earnings[sym] = sorted(events)
            if with_news:
                ctx.news[sym] = repo.get_news(session, sym, datetime(1970, 1, 1), datetime.utcnow() + timedelta(days=1))
        return ctx

    # ------------------------------------------------------------------
    def _next_earnings(self, sym: str, as_of: date) -> Optional[date]:
        for d in self.earnings.get(sym, []):
            if d >= as_of:
                return d
        return None

    def evaluate(self, as_of: Optional[date] = None, previous_scores: Optional[Dict[str, float]] = None,
                 symbols: Optional[List[str]] = None) -> List[ScanRow]:
        """Score and signal every symbol using bars at or before `as_of` (default latest)."""
        symbols = symbols or [s for s in self.universe.symbols if s in self.indicators]
        bench = self.indicators.get(self.universe.benchmark)
        ts = pd.Timestamp(as_of) if as_of else None

        inds: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            ind = self.indicators[sym]
            if ts is not None:
                ind = ind.loc[:ts]
            if ind.empty:
                continue
            inds[sym] = ind
        if not inds:
            return []
        bench_ind = bench.loc[:ts] if (bench is not None and ts is not None) else bench

        # Fundamentals (point-in-time) and universe percentiles
        snaps: Dict[str, FundamentalSnapshot] = {}
        for sym, ind in inds.items():
            d = ind.index[-1].date()
            snaps[sym] = compute_fundamentals_from_frames(sym, self.fundamentals.get(sym, {}), d, float(ind["close"].iloc[-1]),
                                                          self.scoring.fundamentals.max_staleness_days)
        pct = universe_percentiles(snaps, self.scoring.fundamentals.factors)

        rows: List[ScanRow] = []
        for sym, ind in inds.items():
            d = ind.index[-1].date()
            as_of_dt = datetime.combine(d, datetime.max.time())
            patterns = detect_all(ind)
            scfg = self.scoring.sentiment
            window_start = as_of_dt - timedelta(days=scfg.lookback_days)
            articles = [a for a in self.news.get(sym, []) if window_start <= a.published_at <= as_of_dt]
            sent = aggregate_sentiment(articles, as_of_dt, scfg.half_life_days, scfg.min_articles) if self.news else None
            ml_prob = self.ml_predictor(sym, ind, bench_ind) if self.ml_predictor else None
            score = score_symbol(sym, ind, self.scoring, bench_ind, pct.get(sym), snaps[sym].factors, sent, ml_prob,
                                 self.ml_horizon, ml_note=None if ml_prob is not None else (self.ml_status or None))
            prev = (previous_scores or {}).get(sym)
            signal = evaluate_signal(score, patterns, ind, self.signals, self._next_earnings(sym, d), prev)
            latest = {k: (None if pd.isna(v) else float(v)) for k, v in ind.iloc[-1].items()}
            rows.append(ScanRow(sym, d.isoformat(), score, signal, latest, patterns, snaps[sym], sent))
        rows.sort(key=lambda r: r.score.score, reverse=True)
        return rows


def run_scan(session: Session, as_of: Optional[date] = None, persist: bool = True, with_news: bool = True,
             ml_predictor: Optional[MLPredictor] = None, ml_horizon: Optional[int] = None, ml_status: str = "") -> List[ScanRow]:
    ctx = ScanContext.load(session, with_news=with_news, ml_predictor=ml_predictor, ml_horizon=ml_horizon, ml_status=ml_status)
    # previous snapshot scores feed the "score_rising" catalyst
    first_sym = next(iter(ctx.indicators), None)
    ref_date = as_of or (ctx.indicators[first_sym].index[-1].date() if first_sym else date.today())
    prev = {}
    for sym in ctx.universe.symbols:
        p = repo.previous_score(session, sym, ref_date)
        if p:
            prev[sym] = p.score
    rows = ctx.evaluate(as_of, prev)
    if persist:
        for r in rows:
            d = date.fromisoformat(r.as_of)
            repo.save_score(session, symbol=r.symbol, as_of=d, score=r.score.score,
                            components=r.score.to_dict()["components"], explanation=r.score.explanation,
                            indicators=r.indicators, patterns=r.patterns)
            repo.save_signal(session, symbol=r.symbol, as_of=d, signal=r.signal.signal, rule=r.signal.rule,
                             score=r.score.score, reasons=r.signal.reasons, risk_flags=r.signal.risk_flags)
        session.commit()
    return rows
