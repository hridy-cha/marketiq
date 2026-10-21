"""Indicator correctness: hand-computed checks on tiny series plus an independent reference on real data."""

import numpy as np
import pandas as pd
import pytest

from marketiq.analysis import indicators as ta


def _frame(closes, highs=None, lows=None, volumes=None):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": closes.shift(1).fillna(closes),
            "high": pd.Series(highs, index=idx, dtype=float) if highs is not None else closes + 1,
            "low": pd.Series(lows, index=idx, dtype=float) if lows is not None else closes - 1,
            "close": closes,
            "adj_close": closes,
            "volume": pd.Series(volumes, index=idx, dtype=float) if volumes is not None else 1000.0,
        }
    )


def test_sma_hand_computed():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ta.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_matches_recursive_definition():
    s = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
    out = ta.ema(s, 3)
    # pandas ewm(adjust=False) with min_periods seeds with the first value and recurses: alpha = 2/(3+1)
    alpha = 0.5
    e = s.iloc[0]
    for i in range(1, len(s)):
        e = alpha * s.iloc[i] + (1 - alpha) * e
    assert out.iloc[-1] == pytest.approx(e)


def test_rsi_wilder_textbook_example():
    # Standard textbook check: a sequence of 14 gains then evaluate; all-gains gives RSI 100.
    closes = [100 + i for i in range(20)]
    out = ta.rsi(pd.Series(closes, dtype=float), 14)
    assert out.iloc[-1] == pytest.approx(100.0)
    closes_down = [100 - i for i in range(20)]
    out_down = ta.rsi(pd.Series(closes_down, dtype=float), 14)
    assert out_down.iloc[-1] == pytest.approx(0.0)


def test_rsi_manual_wilder_computation():
    rng = np.random.default_rng(0)
    closes = pd.Series(100 + rng.normal(0, 1, 60).cumsum())
    period = 14
    out = ta.rsi(closes, period)
    # Manual Wilder smoothing
    delta = closes.diff()
    gains = delta.clip(lower=0).tolist()
    losses = (-delta.clip(upper=0)).tolist()
    ag = al = None
    rsi_vals = []
    for i in range(1, len(closes)):
        if i < period:
            rsi_vals.append(np.nan)
            continue
        if i == period:
            ag = np.mean(gains[1 : period + 1])
            al = np.mean(losses[1 : period + 1])
        else:
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        rsi_vals.append(100 - 100 / (1 + ag / al))
    # The ewm-with-alpha formulation seeds differently from the SMA seed in the first window but
    # converges quickly; compare the tail.
    assert out.iloc[-1] == pytest.approx(rsi_vals[-1], abs=0.5)


def test_macd_hand_relationship():
    rng = np.random.default_rng(1)
    closes = pd.Series(100 + rng.normal(0, 1, 100).cumsum())
    out = ta.macd(closes)
    expected = closes.ewm(span=12, adjust=False, min_periods=12).mean() - closes.ewm(span=26, adjust=False, min_periods=26).mean()
    pd.testing.assert_series_equal(out["macd"], expected, check_names=False)
    assert (out["macd_hist"].dropna() == (out["macd"] - out["macd_signal"]).dropna()).all()


def test_atr_true_range_hand_computed():
    df = _frame([10, 11, 9], highs=[11, 13, 10], lows=[9, 10, 7])
    tr = ta.true_range(df)
    # bar 2: max(13-10, |13-10|, |10-10|) = 3; bar 3: max(10-7, |10-11|, |7-11|) = 4
    assert tr.iloc[1] == pytest.approx(3.0)
    assert tr.iloc[2] == pytest.approx(4.0)


def test_bollinger_bands_bracket_mid():
    df = _frame(np.linspace(100, 120, 40))
    bb = ta.bollinger(df["close"], 20, 2.0)
    assert (bb["bb_upper"].dropna() >= bb["bb_mid"].dropna()).all()
    assert (bb["bb_lower"].dropna() <= bb["bb_mid"].dropna()).all()


def test_obv_hand_computed():
    df = _frame([10, 11, 10, 10, 12], volumes=[100, 200, 300, 400, 500])
    out = ta.obv(df)
    assert out.tolist() == [0.0, 200.0, -100.0, -100.0, 400.0]


def test_compute_indicators_no_lookahead(aapl_df):
    """Indicator values for a given date must not change when later bars are appended."""
    full = ta.compute_indicators(aapl_df)
    truncated = ta.compute_indicators(aapl_df.iloc[:500])
    cols = ["sma_20", "sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "atr_14", "obv", "volume_ratio", "ret_20"]
    pd.testing.assert_frame_equal(full[cols].iloc[:500], truncated[cols], check_exact=False, rtol=1e-9)


def test_indicators_on_real_aapl_match_reference(aapl_df):
    """Cross-check against straightforward reference formulas on the real AAPL slice."""
    ind = ta.compute_indicators(aapl_df)
    close = aapl_df["close"]
    assert ind["sma_50"].iloc[-1] == pytest.approx(close.iloc[-50:].mean())
    assert ind["sma_200"].iloc[-1] == pytest.approx(close.iloc[-200:].mean())
    assert ind["high_52w"].iloc[-1] == pytest.approx(close.iloc[-252:].max())
    assert 0 <= ind["rsi_14"].iloc[-1] <= 100
    assert ind["volume_ratio"].iloc[-1] == pytest.approx(aapl_df["volume"].iloc[-1] / aapl_df["volume"].iloc[-21:-1].mean())
    # Real December 2024 AAPL closed well above its SMA200; sanity-check the sign without asserting a value.
    assert ind["close"].iloc[-1] > ind["sma_200"].iloc[-1]
    assert ind["rsi_14"].isna().sum() == 14
    assert ind["sma_200"].isna().sum() == 199
