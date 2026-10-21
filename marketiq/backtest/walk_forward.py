"""
Walk-forward validation of the signal strategy.

Pattern adapted from FinRL-X `src/strategies/adaptive_rotation/walk_forward.py` (Apache 2.0): the
history is split into consecutive out-of-sample windows. For each window, anything that is fitted
(the ML direction model) is trained only on data before the window, and the signal strategy is then
backtested inside the window. The windows are stitched into one equity curve (each starts from the
previous ending equity) and reported both per window and in aggregate. Rule parameters are not
optimized per window here; this harness measures how the configured strategy would have performed
without any knowledge of the window it is being tested on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from marketiq.backtest.engine import BacktestConfig, BacktestResult, SignalBacktester
from marketiq.backtest.metrics import compute_metrics
from marketiq.scanner import ScanContext

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window: int
    train_end: str
    test_start: str
    test_end: str
    ml_used: bool
    ml_status: str
    metrics: Dict[str, float]
    n_trades: int


@dataclass
class WalkForwardResult:
    windows: List[WindowResult]
    stitched_equity: pd.Series
    benchmark: Optional[pd.Series]
    aggregate: Dict[str, float]
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "windows": [w.__dict__ for w in self.windows],
            "aggregate": self.aggregate,
            "config": self.config,
            "equity_curve": [{"date": str(i.date()), "equity": float(v),
                              "benchmark": float(self.benchmark[i]) if self.benchmark is not None and i in self.benchmark.index else None}
                             for i, v in self.stitched_equity.items()],
        }


def _fit_model_before(session: Session, train_end: date, horizon: int) -> tuple:
    """Train a direction model on bars up to `train_end`; returns (predictor, horizon, status)."""
    from marketiq.analysis.indicators import compute_indicators
    from marketiq.config.loaders import load_scoring_config, load_universe
    from marketiq.data import repository as repo
    from marketiq.ml.features import build_dataset, make_features
    from marketiq.ml.model import DirectionModel, walk_forward
    from marketiq.ml.registry import validation_gate

    universe = load_universe()
    ind = {}
    for sym in universe.all_symbols:
        df = repo.get_price_frame(session, sym, end=train_end)
        if not df.empty:
            ind[sym] = compute_indicators(df)
    bench = ind.get(universe.benchmark)
    panel = build_dataset({k: v for k, v in ind.items() if k != universe.benchmark}, horizon, 0.0, bench)
    if len(panel) < 500:
        return None, None, f"only {len(panel)} rows before {train_end}; ML skipped"
    try:
        report = walk_forward(panel, horizon, n_folds=3, min_train_rows=500)
    except ValueError as exc:
        return None, None, f"ML validation skipped: {exc}"
    meta = {"name": "wf", "version": str(train_end), "walk_forward": report.to_dict()}
    ok, msg = validation_gate(meta, load_scoring_config().ml)
    if not ok:
        return None, None, msg
    model = DirectionModel(horizon).fit(panel)

    def predict(symbol, ind_df, bench_ind=None):
        feats = make_features(ind_df, bench_ind).iloc[[-1]]
        if feats[model.features].isna().any(axis=1).iloc[0]:
            return None
        return float(model.predict_proba(feats)[0])

    return predict, horizon, msg


def run_walk_forward(
    session: Session,
    start: date,
    end: date,
    window_months: int = 6,
    base_config: Optional[BacktestConfig] = None,
    retrain_ml: bool = True,
    ml_horizon: int = 20,
) -> WalkForwardResult:
    base = base_config or BacktestConfig(start=start, end=end)
    bounds: List[pd.Timestamp] = []
    cur = pd.Timestamp(start)
    while cur < pd.Timestamp(end):
        bounds.append(cur)
        cur = cur + pd.DateOffset(months=window_months)
    bounds.append(pd.Timestamp(end))

    windows: List[WindowResult] = []
    pieces: List[pd.Series] = []
    bench_pieces: List[pd.Series] = []
    capital = base.initial_capital
    for k in range(len(bounds) - 1):
        w_start, w_end = bounds[k].date(), (bounds[k + 1] - pd.Timedelta(days=1)).date()
        train_end = (bounds[k] - pd.Timedelta(days=1)).date()
        predictor, horizon, status = (None, None, "ML retraining disabled")
        if retrain_ml:
            predictor, horizon, status = _fit_model_before(session, train_end, ml_horizon)
        ctx = ScanContext.load(session, end=w_end, with_news=False, ml_predictor=predictor,
                               ml_horizon=horizon, ml_status=status)
        cfg = BacktestConfig(**{**base.__dict__, "start": w_start, "end": w_end, "initial_capital": capital})
        try:
            res: BacktestResult = SignalBacktester(ctx, cfg).run()
        except ValueError as exc:
            logger.warning("window %d skipped: %s", k + 1, exc)
            continue
        windows.append(WindowResult(k + 1, str(train_end), str(w_start), str(w_end), predictor is not None, status,
                                    res.metrics, len(res.trades)))
        pieces.append(res.equity)
        if res.benchmark is not None:
            b = res.benchmark
            bench_pieces.append(b / b.iloc[0] * (bench_pieces[-1].iloc[-1] if bench_pieces else base.initial_capital))
        capital = float(res.equity.iloc[-1])
        logger.info("window %d %s..%s: return %.2f%% trades %d ml=%s", k + 1, w_start, w_end,
                    res.metrics.get("total_return", 0) * 100, len(res.trades), predictor is not None)

    stitched = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    bench = pd.concat(bench_pieces) if bench_pieces else None
    if bench is not None:
        bench = bench[~bench.index.duplicated(keep="last")]
    aggregate = compute_metrics(stitched, base.risk_free_rate, bench) if len(stitched) > 1 else {}
    aggregate["n_windows"] = len(windows)
    aggregate["windows_with_ml"] = sum(1 for w in windows if w.ml_used)
    return WalkForwardResult(windows, stitched, bench, aggregate, {**base.to_dict(), "window_months": window_months, "retrain_ml": retrain_ml})
