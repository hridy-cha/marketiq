"""Every rule in config/signals.yaml is exercised with controlled inputs."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from marketiq.config.loaders import load_signal_config
from marketiq.scoring.scorer import ScoreResult
from marketiq.signals import engine


def _ind(n=260, close=100.0, sma20=95.0, sma50=90.0, sma200=80.0, rsi=60.0, macd_hist=0.5, stretch=1.0, atr_pct=2.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(index=idx)
    df["close"] = close
    df["sma_20"], df["sma_50"], df["sma_200"] = sma20, sma50, sma200
    df["rsi_14"], df["macd_hist"], df["stretch_atr"], df["atr_pct"] = rsi, macd_hist, stretch, atr_pct
    return df


def _patterns(trend="bullish", **recent):
    base = {k: {"direction": "neutral", "recent": False, "reason": f"{k} reason"} for k in
            ["trend_regime", "golden_cross", "death_cross", "breakout_20d", "breakdown_20d", "breakout_52w",
             "macd_bullish_cross", "macd_bearish_cross", "rsi_oversold_reversal"]}
    base["trend_regime"] = {"direction": trend, "recent": True, "reason": f"trend {trend}"}
    for k, v in recent.items():
        base[k]["recent"] = v
    return base


def _score(v: float) -> ScoreResult:
    return ScoreResult("TST", "2024-12-31", v, {}, ["x"])


CFG = load_signal_config()


def test_buy_when_score_and_trend_qualify():
    res = engine.evaluate(_score(75), _patterns(), _ind(), CFG)
    assert res.signal == "BUY" and res.rule == "buy:score_and_trend"
    assert any("buy threshold" in r for r in res.reasons)


def test_buy_blocked_by_trend_becomes_watch():
    res = engine.evaluate(_score(75), _patterns(trend="neutral"), _ind(), CFG)
    assert res.signal == "WATCH" and res.rule == "watch:buy_blocked"
    assert "trend is neutral" in res.reasons[0]


def test_buy_blocked_by_extension_and_rsi():
    res = engine.evaluate(_score(80), _patterns(), _ind(stretch=3.5), CFG)
    assert res.signal == "WATCH" and "extended" in res.reasons[0]
    assert any(f.startswith("extended") for f in res.risk_flags)
    res2 = engine.evaluate(_score(80), _patterns(), _ind(rsi=85), CFG)
    assert res2.signal == "WATCH" and "overbought" in res2.reasons[0]


def test_sell_on_low_score():
    res = engine.evaluate(_score(30), _patterns(), _ind(), CFG)
    assert res.signal == "SELL" and "score_at_or_below_sell" in res.rule


def test_sell_on_death_cross_even_with_good_score():
    res = engine.evaluate(_score(72), _patterns(death_cross=True), _ind(), CFG)
    assert res.signal == "SELL" and "death_cross_recent" in res.rule
    assert any("death_cross reason" in r for r in res.reasons)


def test_sell_on_breakdown():
    res = engine.evaluate(_score(60), _patterns(breakdown_20d=True), _ind(), CFG)
    assert res.signal == "SELL" and "breakdown_recent" in res.rule


def test_sell_on_bearish_trend_with_negative_momentum():
    ind = _ind(close=70, sma20=75, sma50=80, sma200=90, rsi=40, macd_hist=-0.5)
    res = engine.evaluate(_score(45), _patterns(trend="bearish"), ind, CFG)
    assert res.signal == "SELL" and "trend_bearish_and_momentum_down" in res.rule
    # Bearish trend alone (momentum positive) is not a sell.
    ind2 = _ind(close=70, sma20=75, sma50=80, sma200=90, rsi=55, macd_hist=0.5)
    assert engine.evaluate(_score(45), _patterns(trend="bearish"), ind2, CFG).signal == "HOLD"


def test_watch_requires_catalyst_in_band():
    assert engine.evaluate(_score(60), _patterns(), _ind(), CFG).signal == "HOLD"
    res = engine.evaluate(_score(60), _patterns(breakout_20d=True), _ind(), CFG)
    assert res.signal == "WATCH" and res.rule == "watch:catalyst"
    res2 = engine.evaluate(_score(60), _patterns(), _ind(), CFG, previous_score=50.0)
    assert res2.signal == "WATCH" and any("rose +10.0" in r for r in res2.reasons)
    assert engine.evaluate(_score(60), _patterns(), _ind(), CFG, previous_score=55.0).signal == "HOLD"


def test_hold_for_insufficient_history():
    res = engine.evaluate(_score(90), _patterns(), _ind(n=100), CFG)
    assert res.signal == "HOLD" and res.rule == "insufficient_history"


def test_earnings_risk_flag():
    res = engine.evaluate(_score(75), _patterns(), _ind(), CFG, next_earnings=date(2025, 1, 3))
    # last bar of _ind(260) is in Dec 2024; earnings within 7 days flags but does not block the BUY
    assert res.signal == "BUY"
    assert any(f.startswith("earnings in") for f in res.risk_flags)
    assert any("Risk flags present" in r for r in res.reasons)


def test_high_volatility_flag():
    res = engine.evaluate(_score(50), _patterns(), _ind(atr_pct=7.0), CFG)
    assert any("high volatility" in f for f in res.risk_flags)
