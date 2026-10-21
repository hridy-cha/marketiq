"""Backtester mechanics on constructed price paths with scripted signals."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marketiq.backtest.engine import BacktestConfig, SignalBacktester
from marketiq.backtest.metrics import compute_metrics, max_drawdown
from marketiq.config.loaders import Universe, UniverseTicker


def _prices(n=30, start=100.0, step=1.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(start + step * np.arange(n), index=idx)
    return pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
                         "adj_close": close, "volume": 1e6}, index=idx)


def _row(symbol, signal, score, rule="test"):
    return SimpleNamespace(symbol=symbol, score=SimpleNamespace(score=score), signal=SimpleNamespace(signal=signal, rule=rule))


class ScriptedContext:
    """Stands in for ScanContext: evaluate() returns signals from a {date: [rows]} script."""

    def __init__(self, prices, script, benchmark="SPY"):
        self.prices = prices
        self.universe = Universe(benchmark=benchmark, tickers=[UniverseTicker(symbol=s) for s in prices if s != benchmark])
        self.script = script
        self.calls = []

    def evaluate(self, as_of, prev=None):
        self.calls.append(as_of)
        return self.script.get(as_of, [_row(s, "HOLD", 50) for s in self.universe.symbols])


def test_buy_fills_next_open_with_costs_and_sell_realizes_pnl():
    prices = {"AAA": _prices(), "SPY": _prices(start=400)}
    d0 = date(2024, 1, 1)  # first calendar day, rebalance index 0
    d5 = date(2024, 1, 8)  # index 5
    script = {d0: [_row("AAA", "BUY", 80)], d5: [_row("AAA", "SELL", 20, "sell:test")]}
    ctx = ScriptedContext(prices, script)
    cfg = BacktestConfig(start=d0, end=date(2024, 2, 9), initial_capital=10_000, max_positions=1,
                         rebalance_every=5, commission_bps=10, slippage_bps=10, stop_loss_pct=None)
    res = SignalBacktester(ctx, cfg).run()
    buy, sell = res.trades[0], res.trades[1]
    # Decided at the close of 2024-01-01, filled at the open of 2024-01-02 (no same-day information).
    assert buy.date == "2024-01-02" and buy.side == "BUY"
    open_px = prices["AAA"].loc["2024-01-02", "open"]
    assert buy.price == pytest.approx(open_px * (1 + 10 / 1e4))
    assert buy.commission == pytest.approx(buy.notional * 10 / 1e4)
    assert buy.quantity == pytest.approx(buy.notional / buy.price)
    assert sell.date == "2024-01-09" and sell.side == "SELL"
    sell_open = prices["AAA"].loc["2024-01-09", "open"]
    assert sell.price == pytest.approx(sell_open * (1 - 10 / 1e4))
    assert sell.pnl == pytest.approx((sell.price - buy.price) * buy.quantity - sell.commission)
    assert sell.hold_days == 5
    # Cash accounting closes the loop: final equity equals cash since no positions remain.
    assert res.equity.iloc[-1] == pytest.approx(res.cash.iloc[-1])
    assert res.metrics["n_round_trips"] == 1 and res.metrics["trade_win_rate"] == 1.0
    assert ctx.calls[0] == d0 and all(c in prices["AAA"].index.date for c in ctx.calls)


def test_equity_marks_positions_at_close_and_exposure():
    prices = {"AAA": _prices(step=2.0), "SPY": _prices(start=400)}
    d0 = date(2024, 1, 1)
    ctx = ScriptedContext(prices, {d0: [_row("AAA", "BUY", 80)]})
    cfg = BacktestConfig(start=d0, end=date(2024, 2, 9), initial_capital=10_000, max_positions=1, rebalance_every=100,
                         commission_bps=0, slippage_bps=0, stop_loss_pct=None)
    res = SignalBacktester(ctx, cfg).run()
    buy = res.trades[0]
    t = pd.Timestamp("2024-01-10")
    expected = res.cash[t] + buy.quantity * prices["AAA"].loc[t, "close"]
    assert res.equity[t] == pytest.approx(expected)
    assert res.exposure[t] == pytest.approx(buy.quantity * prices["AAA"].loc[t, "close"] / expected)
    assert res.equity.iloc[0] == pytest.approx(10_000)  # nothing held on day 0


def test_stop_loss_triggers_on_close():
    n = 30
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.concatenate([np.full(10, 100.0), np.full(20, 80.0)]), index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "adj_close": close, "volume": 1e6}, index=idx)
    prices = {"AAA": df, "SPY": _prices(start=400)}
    d0 = date(2024, 1, 1)
    ctx = ScriptedContext(prices, {d0: [_row("AAA", "BUY", 80)]})
    cfg = BacktestConfig(start=d0, end=idx[-1].date(), initial_capital=10_000, max_positions=1, rebalance_every=100,
                         commission_bps=0, slippage_bps=0, stop_loss_pct=0.15)
    res = SignalBacktester(ctx, cfg).run()
    sell = res.trades[1]
    assert sell.side == "SELL" and "stop loss" in sell.reason
    # Price drops to 80 at index 10 (2024-01-15); stop seen at that close, filled next open.
    assert sell.date == "2024-01-16" and sell.price == pytest.approx(80.0)


def test_max_positions_and_ranking():
    prices = {s: _prices() for s in ("AAA", "BBB", "CCC")}
    prices["SPY"] = _prices(start=400)
    d0 = date(2024, 1, 1)
    ctx = ScriptedContext(prices, {d0: [_row("AAA", "BUY", 60), _row("BBB", "BUY", 90), _row("CCC", "BUY", 75)]})
    cfg = BacktestConfig(start=d0, end=date(2024, 2, 9), initial_capital=10_000, max_positions=2, rebalance_every=100,
                         commission_bps=0, slippage_bps=0, stop_loss_pct=None)
    res = SignalBacktester(ctx, cfg).run()
    bought = [t.symbol for t in res.trades if t.side == "BUY"]
    assert bought == ["BBB", "CCC"]  # top two scores
    assert all(abs(t.notional - 5_000) < 1 for t in res.trades)  # equal slots of equity / max_positions


def test_metrics_known_series():
    idx = pd.bdate_range("2024-01-01", periods=5)
    eq = pd.Series([100, 110, 99, 120, 120], index=idx, dtype=float)
    m = compute_metrics(eq, risk_free_rate=0.0)
    assert m["total_return"] == pytest.approx(0.20)
    assert max_drawdown(eq) == pytest.approx((99 - 110) / 110)
    assert m["win_rate"] == pytest.approx(2 / 4)
    bench = pd.Series([100, 100, 100, 100, 110], index=idx, dtype=float)
    m2 = compute_metrics(eq, 0.0, bench)
    assert m2["benchmark_total_return"] == pytest.approx(0.10)
    assert m2["excess_return"] == pytest.approx(0.10)
