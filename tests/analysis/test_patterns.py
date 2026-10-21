"""Each detector is exercised on a constructed series where the outcome is known by construction."""

import numpy as np
import pandas as pd

from marketiq.analysis import indicators as ta
from marketiq.analysis import patterns as pt


def _frame_from_close(close, volume=None):
    idx = pd.bdate_range("2022-01-03", periods=len(close))
    close = pd.Series(close, index=idx, dtype=float)
    vol = pd.Series(volume if volume is not None else np.full(len(close), 1_000_000.0), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": vol,
        }
    )
    return ta.compute_indicators(df)


def test_trend_regime_bullish_and_bearish():
    up = _frame_from_close(np.linspace(50, 150, 300))
    assert pt.trend_regime(up).direction == "bullish"
    down = _frame_from_close(np.linspace(150, 50, 300))
    assert pt.trend_regime(down).direction == "bearish"
    short = _frame_from_close(np.linspace(50, 60, 100))
    hit = pt.trend_regime(short)
    assert hit.fired is False and "Not enough history" in hit.reason


def test_golden_and_death_cross_detected():
    # 300 bars down then 300 bars strongly up: SMA50 crosses above SMA200 during the rise.
    close = np.concatenate([np.linspace(150, 80, 300), np.linspace(80, 200, 300)])
    ind = _frame_from_close(close)
    gc = pt.golden_cross(ind)
    assert gc.fired and gc.direction == "bullish" and gc.bars_ago is not None
    dc = pt.death_cross(ind)
    assert dc.fired is False or dc.bars_ago > gc.bars_ago  # the most recent event is the golden cross


def test_breakout_20d_requires_volume_confirmation():
    base = np.full(60, 100.0)
    close = base.copy()
    close[-1] = 110.0  # above prior 20-day high
    vol = np.full(60, 1_000_000.0)
    no_vol = _frame_from_close(close, vol)
    assert pt.breakout_20d(no_vol).fired is False
    vol[-1] = 3_000_000.0
    with_vol = _frame_from_close(close, vol)
    hit = pt.breakout_20d(with_vol)
    assert hit.fired and hit.bars_ago == 0 and "3.0x" in hit.reason


def test_breakdown_20d():
    close = np.full(60, 100.0)
    close[-1] = 90.0
    vol = np.full(60, 1_000_000.0)
    vol[-1] = 2_000_000.0
    hit = pt.breakdown_20d(_frame_from_close(close, vol))
    assert hit.fired and hit.direction == "bearish" and hit.bars_ago == 0


def test_breakout_52w():
    close = np.concatenate([np.full(300, 100.0), [101.0]])
    hit = pt.breakout_52w(_frame_from_close(close))
    assert hit.fired and hit.bars_ago == 0


def test_macd_cross_detectors():
    close = np.concatenate([np.linspace(100, 80, 60), np.linspace(80, 120, 60)])
    ind = _frame_from_close(close)
    bull = pt.macd_bullish_cross(ind)
    assert bull.fired and bull.direction == "bullish"


def test_rsi_oversold_reversal():
    # Sharp drop (RSI goes well below 30) then a rebound (RSI rises back above 30).
    close = np.concatenate([np.full(30, 100.0), np.linspace(100, 60, 20), np.linspace(60, 75, 10)])
    ind = _frame_from_close(close)
    assert ind["rsi_14"].iloc[49] < 30
    hit = pt.rsi_oversold_reversal(ind)
    assert hit.fired and hit.direction == "bullish"


def test_mean_reversion_stretch_flags_extension():
    close = np.concatenate([np.full(40, 100.0) + np.random.default_rng(0).normal(0, 0.3, 40), [130.0]])
    hit = pt.mean_reversion_stretch(_frame_from_close(close))
    assert hit.fired and hit.direction == "bearish" and "Extended" in hit.reason


def test_detect_all_on_real_data_has_every_detector(aapl_df):
    ind = ta.compute_indicators(aapl_df)
    out = pt.detect_all(ind)
    assert set(out) == {fn.__name__ if fn.__name__ != "mean_reversion_stretch" else "stretch" for fn in pt.DETECTORS}
    for v in out.values():
        assert set(v) >= {"fired", "direction", "reason", "recent"}
        assert v["direction"] in {"bullish", "bearish", "neutral"}
