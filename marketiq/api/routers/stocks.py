"""Universe, rankings and per-stock detail endpoints."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from marketiq.analysis.fundamentals import compute_fundamentals
from marketiq.analysis.indicators import compute_indicators
from marketiq.analysis.patterns import detect_all
from marketiq.analysis.sentiment import aggregate_sentiment
from marketiq.api.deps import db
from marketiq.config.loaders import cached_universe, load_scoring_config
from marketiq.data import repository as repo

router = APIRouter(prefix="/api", tags=["stocks"])

INDICATOR_SERIES = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "rsi_14", "macd", "macd_signal", "macd_hist",
                    "atr_14", "bb_upper", "bb_mid", "bb_lower", "obv", "volume_sma_20", "volume_ratio"]


def _clean(v):
    if v is None:
        return None
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return v


def _symbol_or_404(symbol: str) -> str:
    s = symbol.upper()
    u = cached_universe()
    if s not in u.all_symbols:
        raise HTTPException(404, f"{s} is not in the universe")
    return s


@router.get("/universe")
def universe():
    u = cached_universe()
    return {"benchmark": u.benchmark, "tickers": [t.model_dump() for t in u.tickers]}


@router.get("/rankings")
def rankings(session: Session = Depends(db), as_of: Optional[date] = None):
    scores = repo.latest_scores(session, as_of)
    if not scores:
        return {"as_of": None, "rows": [], "message": "No scan snapshots yet. Run `marketiq scan`."}
    sigs = repo.latest_signals(session, scores[0].as_of)
    u = {t.symbol: t for t in cached_universe().tickers}
    rows = []
    for rank, s in enumerate(scores, 1):
        sig = sigs.get(s.symbol)
        ind = s.indicators or {}
        rows.append({
            "rank": rank,
            "symbol": s.symbol,
            "name": u[s.symbol].name if s.symbol in u else "",
            "sector": u[s.symbol].sector if s.symbol in u else "",
            "score": s.score,
            "components": {k: v.get("score") for k, v in (s.components or {}).items()},
            "signal": sig.signal if sig else None,
            "rule": sig.rule if sig else None,
            "risk_flags": sig.risk_flags if sig else [],
            "close": ind.get("close"),
            "ret_1": ind.get("ret_1"),
            "ret_20": ind.get("ret_20"),
            "rsi_14": ind.get("rsi_14"),
            "trend": (s.patterns or {}).get("trend_regime", {}).get("direction"),
            "recent_patterns": [k for k, v in (s.patterns or {}).items() if v.get("recent") and k != "trend_regime"],
        })
    return {"as_of": scores[0].as_of.isoformat(), "rows": rows}


@router.get("/stocks/{symbol}/bars")
def bars(symbol: str, session: Session = Depends(db), start: Optional[date] = None, end: Optional[date] = None,
         limit: int = Query(750, le=5000)):
    s = _symbol_or_404(symbol)
    df = repo.get_price_frame(session, s, start, end)
    if df.empty:
        raise HTTPException(404, f"No price data for {s}")
    df = df.tail(limit)
    return {"symbol": s, "bars": [
        {"date": i.strftime("%Y-%m-%d"), "open": _clean(r.open), "high": _clean(r.high), "low": _clean(r.low),
         "close": _clean(r.close), "adj_close": _clean(r.adj_close), "volume": _clean(r.volume)}
        for i, r in df.iterrows()]}


@router.get("/stocks/{symbol}/indicators")
def indicators(symbol: str, session: Session = Depends(db), limit: int = Query(750, le=5000)):
    s = _symbol_or_404(symbol)
    df = repo.get_price_frame(session, s)
    if df.empty:
        raise HTTPException(404, f"No price data for {s}")
    ind = compute_indicators(df)
    tail = ind.tail(limit)
    series: Dict[str, List] = {col: [_clean(v) for v in tail[col]] for col in INDICATOR_SERIES if col in tail}
    latest = {k: _clean(v) for k, v in ind.iloc[-1].items()}
    return {"symbol": s, "as_of": ind.index[-1].strftime("%Y-%m-%d"),
            "dates": [i.strftime("%Y-%m-%d") for i in tail.index], "series": series, "latest": latest,
            "patterns": detect_all(ind)}


@router.get("/stocks/{symbol}/fundamentals")
def fundamentals(symbol: str, session: Session = Depends(db), as_of: Optional[date] = None):
    s = _symbol_or_404(symbol)
    df = repo.get_price_frame(session, s, end=as_of)
    if df.empty:
        raise HTTPException(404, f"No price data for {s}")
    d = df.index[-1].date()
    payloads = repo.get_fundamental_payloads(session, s, as_of=d)
    snap = compute_fundamentals(s, payloads, d, float(df["close"].iloc[-1]), load_scoring_config().fundamentals.max_staleness_days)
    nxt = repo.next_earnings(session, s, d)
    out = snap.to_dict()
    out["next_earnings"] = nxt.event_date.isoformat() if nxt else None
    # Last 8 quarters of headline figures for the UI table.
    income = [p for p in payloads if p.statement == "income"][-8:]
    out["quarters"] = [{"period_end": p.period_end.isoformat(), "filing_date": p.filing_date.isoformat() if p.filing_date else None,
                        "revenue": _clean(p.data.get("revenue")), "net_income": _clean(p.data.get("netIncome")),
                        "eps": _clean(p.data.get("epsdiluted", p.data.get("eps")))} for p in income]
    return out


@router.get("/stocks/{symbol}/news")
def news(symbol: str, session: Session = Depends(db), days: int = Query(30, le=365)):
    s = _symbol_or_404(symbol)
    end = datetime.utcnow() + timedelta(days=1)
    rows = repo.get_news(session, s, end - timedelta(days=days), end)
    cfg = load_scoring_config().sentiment
    agg = aggregate_sentiment(rows, datetime.utcnow(), cfg.half_life_days, cfg.min_articles)
    return {"symbol": s, "aggregate": agg.__dict__, "articles": [
        {"published_at": a.published_at.isoformat(), "title": a.title, "summary": a.summary[:400], "url": a.url,
         "publisher": a.publisher, "sentiment_label": a.sentiment_label, "sentiment_score": a.sentiment_score,
         "source": a.source} for a in rows]}


@router.get("/stocks/{symbol}/score")
def score(symbol: str, session: Session = Depends(db), as_of: Optional[date] = None):
    s = _symbol_or_404(symbol)
    snaps = repo.latest_scores(session, as_of)
    snap = next((x for x in snaps if x.symbol == s), None)
    if snap is None:
        raise HTTPException(404, f"No score snapshot for {s}. Run `marketiq scan`.")
    sig = repo.latest_signals(session, snap.as_of).get(s)
    history = repo.score_history(session, s, limit=120)
    return {"symbol": s, "as_of": snap.as_of.isoformat(), "score": snap.score, "components": snap.components,
            "explanation": snap.explanation, "patterns": snap.patterns, "indicators": snap.indicators,
            "signal": {"signal": sig.signal, "rule": sig.rule, "reasons": sig.reasons, "risk_flags": sig.risk_flags} if sig else None,
            "history": [{"as_of": h.as_of.isoformat(), "score": h.score} for h in history]}


@router.get("/signals")
def signals(session: Session = Depends(db), as_of: Optional[date] = None, signal: Optional[str] = None):
    sigs = repo.latest_signals(session, as_of)
    rows = [{"symbol": s.symbol, "as_of": s.as_of.isoformat(), "signal": s.signal, "rule": s.rule, "score": s.score,
             "reasons": s.reasons, "risk_flags": s.risk_flags} for s in sigs.values()]
    if signal:
        rows = [r for r in rows if r["signal"] == signal.upper()]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"rows": rows}
