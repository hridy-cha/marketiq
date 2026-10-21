"""
Event-driven backtester for the MarketIQ signal strategy.

Strategy (long only):
- On each rebalance date (every `rebalance_every` trading days) the universe is evaluated with the
  scan context using only bars up to and including that date.
- Open positions are closed when their signal is SELL, when the composite score falls below
  `exit_score`, when a stop loss is hit, or when `max_hold_days` elapses.
- Free slots (up to `max_positions`) are filled with the highest-scoring BUY (and optionally WATCH)
  names not already held.
- Orders are generated at the close of the decision date and filled at the next bar's open, with
  slippage and commission in basis points. There is no same-day information in any fill.
- Equity is marked to market at every close. Cash earns nothing.

The result holds the equity curve, every trade with its reason, and metrics versus the benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from marketiq.backtest.metrics import compute_metrics
from marketiq.scanner import ScanContext, ScanRow

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start: date
    end: date
    initial_capital: float = 100_000.0
    max_positions: int = 8
    rebalance_every: int = 5  # trading days between evaluations
    commission_bps: float = 5.0  # per side, on notional
    slippage_bps: float = 5.0  # adverse move applied to the open price
    exit_score: float = 45.0  # close a position when its score falls below this
    stop_loss_pct: Optional[float] = 0.15  # close if price falls this far below entry (None disables)
    max_hold_days: Optional[int] = None  # trading days; None = no limit
    enter_on: tuple = ("BUY",)  # signals that open a position
    size_by_score: bool = False  # weight slots by score instead of equally
    risk_free_rate: float = 0.02

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = self.start.isoformat(), self.end.isoformat()
        d["enter_on"] = list(self.enter_on)
        return d


@dataclass
class Trade:
    symbol: str
    side: str  # BUY | SELL
    date: str
    price: float
    quantity: float
    notional: float
    commission: float
    reason: str
    pnl: Optional[float] = None  # realized on SELL
    hold_days: Optional[int] = None


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_index: int


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series
    cash: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    benchmark: Optional[pd.Series]
    exposure: pd.Series  # fraction of equity invested
    signals_log: List[dict] = field(default_factory=list)

    def to_dict(self, include_curves: bool = True) -> dict:
        d = {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "trades": [asdict(t) for t in self.trades],
            "n_trades": len(self.trades),
        }
        if include_curves:
            d["equity_curve"] = [{"date": str(i.date()), "equity": float(v), "cash": float(self.cash[i]),
                                  "exposure": float(self.exposure[i]),
                                  "benchmark": (float(self.benchmark[i]) if self.benchmark is not None and i in self.benchmark.index else None)}
                                 for i, v in self.equity.items()]
        return d


class SignalBacktester:
    def __init__(self, ctx: ScanContext, config: BacktestConfig):
        self.ctx = ctx
        self.cfg = config
        bench = ctx.prices.get(ctx.universe.benchmark)
        # Master calendar: union of trading days across universe prices within the window.
        idx = None
        for sym in ctx.universe.symbols:
            p = ctx.prices.get(sym)
            if p is None:
                continue
            i = p.loc[pd.Timestamp(config.start):pd.Timestamp(config.end)].index
            idx = i if idx is None else idx.union(i)
        if idx is None or len(idx) == 0:
            raise ValueError("No price data in the backtest window")
        self.calendar = idx
        self.bench = bench["close"].reindex(idx).ffill() if bench is not None else None

    # ------------------------------------------------------------------
    def _open_price(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        p = self.ctx.prices[sym]
        if ts not in p.index:
            return None
        v = p.at[ts, "open"]
        return None if pd.isna(v) else float(v)

    def _close_price(self, sym: str, ts: pd.Timestamp) -> Optional[float]:
        p = self.ctx.prices[sym]
        row = p.loc[:ts]
        if row.empty:
            return None
        v = row["close"].iloc[-1]
        return None if pd.isna(v) else float(v)

    def run(self) -> BacktestResult:
        cfg = self.cfg
        cash = cfg.initial_capital
        positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        pending: List[dict] = []  # orders decided at close, filled next open
        equity_curve, cash_curve, exposure_curve = [], [], []
        signals_log: List[dict] = []
        prev_scores: Dict[str, float] = {}

        for i, ts in enumerate(self.calendar):
            # 1. Fill pending orders at today's open.
            for order in pending:
                sym = order["symbol"]
                px = self._open_price(sym, ts)
                if px is None:
                    continue
                if order["side"] == "SELL":
                    pos = positions.pop(sym, None)
                    if pos is None:
                        continue
                    fill = px * (1 - cfg.slippage_bps / 1e4)
                    notional = pos.quantity * fill
                    commission = notional * cfg.commission_bps / 1e4
                    cash += notional - commission
                    pnl = (fill - pos.entry_price) * pos.quantity - commission - order.get("entry_commission", 0.0)
                    trades.append(Trade(sym, "SELL", str(ts.date()), fill, pos.quantity, notional, commission,
                                        order["reason"], pnl=pnl, hold_days=i - pos.entry_index))
                else:
                    fill = px * (1 + cfg.slippage_bps / 1e4)
                    budget = order["notional"]
                    commission = budget * cfg.commission_bps / 1e4
                    if budget + commission > cash or budget <= 0:
                        budget = max(0.0, cash / (1 + cfg.commission_bps / 1e4))
                        commission = budget * cfg.commission_bps / 1e4
                    qty = budget / fill
                    if qty <= 0:
                        continue
                    cash -= budget + commission
                    positions[sym] = Position(sym, qty, fill, ts, i)
                    trades.append(Trade(sym, "BUY", str(ts.date()), fill, qty, budget, commission, order["reason"]))
            pending = []

            # 2. Mark to market at the close.
            invested = 0.0
            for sym, pos in positions.items():
                c = self._close_price(sym, ts)
                invested += pos.quantity * (c if c is not None else pos.entry_price)
            equity = cash + invested
            equity_curve.append(equity)
            cash_curve.append(cash)
            exposure_curve.append(invested / equity if equity > 0 else 0.0)

            # 3. Decide at the close on rebalance days (and always check stops).
            is_rebalance = (i % cfg.rebalance_every == 0)
            decisions: Dict[str, dict] = {}
            if is_rebalance and i < len(self.calendar) - 1:
                rows = self.ctx.evaluate(ts.date(), prev_scores)
                by_sym = {r.symbol: r for r in rows}
                prev_scores = {r.symbol: r.score.score for r in rows}
                signals_log.append({"date": str(ts.date()), "signals": {r.symbol: r.signal.signal for r in rows},
                                    "scores": {r.symbol: r.score.score for r in rows}})
                # exits
                for sym, pos in list(positions.items()):
                    r = by_sym.get(sym)
                    if r is None:
                        continue
                    reason = None
                    if r.signal.signal == "SELL":
                        reason = f"SELL signal ({r.signal.rule})"
                    elif r.score.score < cfg.exit_score:
                        reason = f"score {r.score.score:.1f} below exit threshold {cfg.exit_score:.0f}"
                    elif cfg.max_hold_days and (i - pos.entry_index) >= cfg.max_hold_days:
                        reason = f"max hold {cfg.max_hold_days} days reached"
                    if reason:
                        decisions[sym] = {"symbol": sym, "side": "SELL", "reason": reason}
                # entries
                remaining = cfg.max_positions - (len(positions) - sum(1 for d in decisions.values() if d["side"] == "SELL"))
                candidates = [r for r in rows if r.signal.signal in cfg.enter_on and r.symbol not in positions]
                candidates.sort(key=lambda r: r.score.score, reverse=True)
                chosen = candidates[:max(0, remaining)]
                if chosen:
                    # Capital available after planned exits, split across the open slots.
                    projected_cash = cash + sum(
                        positions[s].quantity * (self._close_price(s, ts) or positions[s].entry_price)
                        for s, d in decisions.items() if d["side"] == "SELL")
                    slot_value = equity / cfg.max_positions
                    weights = [max(r.score.score, 1.0) for r in chosen] if cfg.size_by_score else [1.0] * len(chosen)
                    total_w = sum(weights)
                    cash_after_fees = projected_cash / (1 + cfg.commission_bps / 1e4)
                    for r, w in zip(chosen, weights):
                        notional = min(slot_value * len(chosen) * (w / total_w), cash_after_fees * (w / total_w))
                        decisions[r.symbol] = {"symbol": r.symbol, "side": "BUY", "notional": notional,
                                               "reason": f"{r.signal.signal} score {r.score.score:.1f} ({r.signal.rule})"}
            # stop losses checked daily on the close
            if cfg.stop_loss_pct:
                for sym, pos in positions.items():
                    if sym in decisions:
                        continue
                    c = self._close_price(sym, ts)
                    if c is not None and c <= pos.entry_price * (1 - cfg.stop_loss_pct):
                        decisions[sym] = {"symbol": sym, "side": "SELL",
                                          "reason": f"stop loss: {c:.2f} is {cfg.stop_loss_pct:.0%} below entry {pos.entry_price:.2f}"}
            pending = list(decisions.values())

        idx = self.calendar
        equity_s = pd.Series(equity_curve, index=idx, name="equity")
        cash_s = pd.Series(cash_curve, index=idx, name="cash")
        exposure_s = pd.Series(exposure_curve, index=idx, name="exposure")
        bench_curve = None
        if self.bench is not None and self.bench.notna().any():
            b = self.bench.dropna()
            bench_curve = b / b.iloc[0] * cfg.initial_capital
        metrics = compute_metrics(equity_s, cfg.risk_free_rate, bench_curve)
        closed = [t for t in trades if t.side == "SELL"]
        if closed:
            wins = [t for t in closed if (t.pnl or 0) > 0]
            metrics["n_round_trips"] = len(closed)
            metrics["trade_win_rate"] = len(wins) / len(closed)
            metrics["avg_trade_pnl"] = float(np.mean([t.pnl for t in closed]))
            metrics["avg_hold_days"] = float(np.mean([t.hold_days for t in closed]))
            gross_win = sum(t.pnl for t in wins)
            gross_loss = -sum(t.pnl for t in closed if (t.pnl or 0) <= 0)
            metrics["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        metrics["avg_exposure"] = float(exposure_s.mean())
        metrics["total_commission"] = float(sum(t.commission for t in trades))
        return BacktestResult(cfg, equity_s, cash_s, trades, metrics, bench_curve, exposure_s, signals_log)
