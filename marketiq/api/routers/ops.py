"""Watchlist, scan trigger, ML, backtests, paper trading and config endpoints."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from marketiq import __version__
from marketiq.api.deps import db
from marketiq.config.loaders import cached_universe, load_scoring_config, load_signal_config
from marketiq.config.settings import get_settings
from marketiq.data import repository as repo
from marketiq.data.models import BacktestRun, MLModelRecord

router = APIRouter(prefix="/api", tags=["ops"])


@router.get("/health")
def health(session: Session = Depends(db)):
    cov = repo.price_coverage(session)
    scores = repo.latest_scores(session)
    return {"status": "ok", "version": __version__, "symbols_with_prices": int(len(cov)),
            "latest_bar": str(cov["last"].max()) if not cov.empty else None,
            "latest_scan": scores[0].as_of.isoformat() if scores else None,
            "live_trading_enabled": get_settings().enable_live_trading}


@router.get("/config")
def config():
    sc = load_scoring_config()
    sg = load_signal_config()
    return {"scoring": sc.model_dump(), "signals": sg.model_dump(),
            "files": {"scoring": str(get_settings().data.path("scoring_file")), "signals": str(get_settings().data.path("signals_file"))}}


class WeightsUpdate(BaseModel):
    weights: dict = Field(..., description="component -> weight; normalized on save")


@router.put("/config/weights")
def update_weights(body: WeightsUpdate):
    """Persist new component weights to config/scoring.yaml. Takes effect on the next scan."""
    import yaml

    from marketiq.config.loaders import ScoringConfig

    path = get_settings().data.path("scoring_file")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["weights"] = {k: float(v) for k, v in body.weights.items()}
    try:
        cfg = ScoringConfig(**raw)  # validates names and positivity
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return {"weights": cfg.weights}


# ---------------------------------------------------------------- watchlist
@router.get("/watchlist")
def watchlist(session: Session = Depends(db)):
    items = repo.list_watchlist(session)
    scores = {s.symbol: s for s in repo.latest_scores(session)}
    sigs = repo.latest_signals(session)
    return {"items": [{"symbol": i.symbol, "note": i.note, "added_by": i.added_by, "created_at": i.created_at.isoformat(),
                       "score": scores[i.symbol].score if i.symbol in scores else None,
                       "signal": sigs[i.symbol].signal if i.symbol in sigs else None,
                       "close": (scores[i.symbol].indicators or {}).get("close") if i.symbol in scores else None}
                      for i in items]}


class WatchlistAdd(BaseModel):
    symbol: str
    note: str = ""


@router.post("/watchlist", status_code=201)
def watchlist_add(body: WatchlistAdd, session: Session = Depends(db)):
    sym = body.symbol.upper()
    if sym not in cached_universe().all_symbols:
        raise HTTPException(404, f"{sym} is not in the universe")
    item = repo.add_watchlist(session, sym, body.note)
    return {"symbol": item.symbol, "note": item.note}


@router.delete("/watchlist/{symbol}")
def watchlist_remove(symbol: str, session: Session = Depends(db)):
    if not repo.remove_watchlist(session, symbol.upper()):
        raise HTTPException(404, "not on watchlist")
    return {"removed": symbol.upper()}


# ---------------------------------------------------------------- scan / ml
@router.post("/scan")
def trigger_scan(session: Session = Depends(db), as_of: Optional[date] = None):
    """Run a scan synchronously (takes a few seconds for a 30-stock universe)."""
    from marketiq.ml.registry import load_predictor
    from marketiq.scanner import run_scan

    predictor, horizon, status = load_predictor()
    rows = run_scan(session, as_of, persist=True, ml_predictor=predictor, ml_horizon=horizon, ml_status=status)
    return {"as_of": rows[0].as_of if rows else None, "n": len(rows), "ml_status": status,
            "signals": {r.symbol: r.signal.signal for r in rows}}


@router.get("/ml/models")
def ml_models(session: Session = Depends(db)):
    from marketiq.ml.registry import load_latest, validation_gate

    records = session.scalars(select(MLModelRecord).order_by(MLModelRecord.created_at.desc())).all()
    _, meta = load_latest()
    gate = validation_gate(meta) if meta else (False, "No trained model")
    return {"latest": meta, "gate": {"passed": gate[0], "message": gate[1]},
            "records": [{"name": r.name, "version": r.version, "horizon_days": r.horizon_days, "metrics": r.metrics,
                         "train_start": r.train_start.isoformat() if r.train_start else None,
                         "train_end": r.train_end.isoformat() if r.train_end else None,
                         "created_at": r.created_at.isoformat()} for r in records]}


@router.get("/ml/predictions")
def ml_predictions(session: Session = Depends(db)):
    """Current probabilities from the latest model, gate status included (predictions are shown even
    when the gate fails, clearly labelled, so the model can be inspected)."""
    from marketiq.analysis.indicators import compute_indicators
    from marketiq.ml.registry import load_predictor

    predictor, horizon, status = load_predictor(enforce_gate=False)
    _, _, gated_status = load_predictor()
    if predictor is None:
        return {"available": False, "status": status, "rows": []}
    u = cached_universe()
    bench = compute_indicators(repo.get_price_frame(session, u.benchmark))
    rows = []
    for sym in u.symbols:
        df = repo.get_price_frame(session, sym)
        if df.empty:
            continue
        ind = compute_indicators(df)
        p = predictor(sym, ind, bench)
        rows.append({"symbol": sym, "as_of": ind.index[-1].strftime("%Y-%m-%d"), "prob_up": p})
    rows.sort(key=lambda r: -(r["prob_up"] or 0))
    return {"available": True, "horizon_days": horizon, "status": status, "gate_status": gated_status,
            "used_in_scoring": "passed" in gated_status, "rows": rows}


# ---------------------------------------------------------------- backtests
class BacktestRequest(BaseModel):
    start: date = date(2021, 1, 4)
    end: date = date(2025, 10, 15)
    initial_capital: float = 100_000.0
    max_positions: int = 8
    rebalance_every: int = 5
    exit_score: float = 45.0
    stop_loss_pct: Optional[float] = 0.15
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    enter_on_watch: bool = False
    use_ml: bool = True
    name: str = "signal-strategy"


def _run_backtest(req: BacktestRequest) -> int:
    from marketiq.backtest.engine import BacktestConfig, SignalBacktester
    from marketiq.data.db import session_scope
    from marketiq.ml.registry import load_predictor
    from marketiq.scanner import ScanContext

    predictor, horizon, status = load_predictor() if req.use_ml else (None, None, "ML disabled for this run")
    cfg = BacktestConfig(start=req.start, end=req.end, initial_capital=req.initial_capital, max_positions=req.max_positions,
                         rebalance_every=req.rebalance_every, exit_score=req.exit_score, stop_loss_pct=req.stop_loss_pct or None,
                         commission_bps=req.commission_bps, slippage_bps=req.slippage_bps,
                         enter_on=("BUY", "WATCH") if req.enter_on_watch else ("BUY",))
    with session_scope() as s:
        ctx = ScanContext.load(s, end=cfg.end, with_news=False, ml_predictor=predictor, ml_horizon=horizon, ml_status=status)
        res = SignalBacktester(ctx, cfg).run()
        run = BacktestRun(name=req.name, kind="signal", start_date=cfg.start, end_date=cfg.end,
                          config={**cfg.to_dict(), "ml_status": status}, metrics=res.metrics,
                          equity_curve=res.to_dict()["equity_curve"], trades=[t.__dict__ for t in res.trades])
        s.add(run)
        s.flush()
        return run.id


@router.post("/backtests", status_code=201)
def create_backtest(req: BacktestRequest):
    """Runs synchronously; a 4-year weekly backtest takes about 1-2 minutes."""
    run_id = _run_backtest(req)
    return {"id": run_id}


@router.get("/backtests")
def list_backtests(session: Session = Depends(db)):
    runs = session.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc())).all()
    return {"runs": [{"id": r.id, "name": r.name, "kind": r.kind, "start_date": r.start_date.isoformat(),
                      "end_date": r.end_date.isoformat(), "metrics": r.metrics, "created_at": r.created_at.isoformat(),
                      "n_trades": len(r.trades or [])} for r in runs]}


@router.get("/backtests/{run_id}")
def get_backtest(run_id: int, session: Session = Depends(db)):
    r = session.get(BacktestRun, run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return {"id": r.id, "name": r.name, "kind": r.kind, "start_date": r.start_date.isoformat(), "end_date": r.end_date.isoformat(),
            "config": r.config, "metrics": r.metrics, "equity_curve": r.equity_curve, "trades": r.trades,
            "created_at": r.created_at.isoformat()}


# ---------------------------------------------------------------- paper trading
class PaperCreate(BaseModel):
    name: str = "default"
    starting_cash: float = 100_000.0
    reset: bool = False


@router.post("/paper/accounts", status_code=201)
def paper_create(body: PaperCreate, session: Session = Depends(db)):
    from marketiq.paper.broker import PaperBroker

    acct = PaperBroker.reset_account(session, body.name, body.starting_cash) if body.reset else PaperBroker.create_account(session, body.name, body.starting_cash)
    return {"name": acct.name, "cash": acct.cash, "starting_cash": acct.starting_cash}


@router.get("/paper/accounts/{name}")
def paper_portfolio(name: str, session: Session = Depends(db)):
    from marketiq.paper.broker import PaperBroker, PaperTradingError

    try:
        b = PaperBroker(session, name)
    except PaperTradingError as exc:
        raise HTTPException(404, str(exc))
    pf = b.portfolio()
    pf["trades"] = [{"symbol": t.symbol, "side": t.side, "quantity": t.quantity, "price": t.price, "commission": t.commission,
                     "trade_date": t.trade_date.isoformat(), "reason": t.reason} for t in b.trades()]
    pf["equity_history"] = [{"as_of": e.as_of.isoformat(), "equity": e.equity, "cash": e.cash} for e in b.equity_history()]
    return pf


class PaperOrder(BaseModel):
    symbol: str
    side: str  # BUY | SELL
    notional: Optional[float] = None
    quantity: Optional[float] = None
    reason: str = "manual"


@router.post("/paper/accounts/{name}/orders", status_code=201)
def paper_order(name: str, body: PaperOrder, session: Session = Depends(db)):
    from marketiq.paper.broker import PaperBroker, PaperTradingError

    sym = body.symbol.upper()
    if sym not in cached_universe().all_symbols:
        raise HTTPException(404, f"{sym} is not in the universe")
    try:
        b = PaperBroker(session, name)
        td = repo.latest_price_date(session, sym)
        fill = b.buy(sym, body.notional, body.quantity, trade_date=td, reason=body.reason) if body.side.upper() == "BUY" \
            else b.sell(sym, body.quantity, trade_date=td, reason=body.reason)
        b.mark(td)
    except PaperTradingError as exc:
        raise HTTPException(400, str(exc))
    return fill.__dict__


@router.post("/paper/accounts/{name}/rebalance")
def paper_rebalance(name: str, session: Session = Depends(db), max_positions: int = 8, exit_score: float = 45.0):
    """Apply the latest persisted scan to the paper account."""
    from marketiq.paper.broker import PaperBroker, PaperTradingError
    from marketiq.scanner import ScanContext

    scores = repo.latest_scores(session)
    if not scores:
        raise HTTPException(400, "No scan snapshots. Run a scan first.")
    try:
        b = PaperBroker(session, name)
    except PaperTradingError as exc:
        raise HTTPException(404, str(exc))
    # Re-evaluate from stored data so reasons match the snapshot date.
    ctx = ScanContext.load(session, end=scores[0].as_of, with_news=False)
    rows = ctx.evaluate(scores[0].as_of)
    fills = b.apply_signals(rows, max_positions=max_positions, exit_score=exit_score, trade_date=scores[0].as_of)
    return {"as_of": scores[0].as_of.isoformat(), "fills": [f.__dict__ for f in fills], "portfolio": b.portfolio(scores[0].as_of)}
