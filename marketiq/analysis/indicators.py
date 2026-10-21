"""
Technical indicators in plain pandas. Every function takes a price frame indexed by date with
columns open, high, low, close, volume (adj_close optional) and returns Series/DataFrames aligned
to that index. No look-ahead: each value uses only bars at or before its own date.

Conventions:
- RSI uses Wilder's smoothing (the standard "RSI 14").
- MACD uses EMA(12) - EMA(26) with a 9-period EMA signal line.
- ATR uses Wilder's smoothing of true range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Returns NaN for the first `period` bars."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there were no losses in the window, RSI is 100 by definition.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std})


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def realized_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    r = np.log(close / close.shift(1))
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    return vol * np.sqrt(252) if annualize else vol


def compute_indicators(
    df: pd.DataFrame,
    sma_windows=(20, 50, 200),
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_window: int = 20,
    return_windows=(1, 5, 20, 60),
) -> pd.DataFrame:
    """All indicators as columns alongside the input OHLCV. Uses `close` (not adj_close) so values
    match what a chart shows; returns use adj_close when available so splits do not distort them."""
    if df.empty:
        return df.copy()
    out = df.copy()
    close = out["close"].astype(float)
    ret_base = out["adj_close"].astype(float) if "adj_close" in out.columns and out["adj_close"].notna().any() else close

    for w in sma_windows:
        out[f"sma_{w}"] = sma(close, w)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out[f"rsi_{rsi_period}"] = rsi(close, rsi_period)
    out = out.join(macd(close))
    out[f"atr_{atr_period}"] = atr(out, atr_period)
    out["atr_pct"] = out[f"atr_{atr_period}"] / close * 100.0
    out = out.join(bollinger(close))
    out["obv"] = obv(out)
    out["obv_sma_20"] = sma(out["obv"], 20)
    out[f"volume_sma_{volume_window}"] = sma(out["volume"].astype(float), volume_window)
    # Today's volume relative to the average of the *prior* `volume_window` bars, so a single
    # high-volume day is compared against normal activity rather than diluting its own baseline.
    out["volume_ratio"] = out["volume"] / out[f"volume_sma_{volume_window}"].shift(1)
    out["realized_vol_20"] = realized_volatility(ret_base, 20)
    for w in return_windows:
        out[f"ret_{w}"] = ret_base.pct_change(w)
    out["high_52w"] = close.rolling(252, min_periods=126).max()
    out["low_52w"] = close.rolling(252, min_periods=126).min()
    out["pct_from_52w_high"] = (close / out["high_52w"] - 1.0) * 100.0
    out["pct_from_52w_low"] = (close / out["low_52w"] - 1.0) * 100.0
    out["high_20"] = close.rolling(20, min_periods=20).max()
    out["low_20"] = close.rolling(20, min_periods=20).min()
    out["drawdown_from_high_52w"] = out["pct_from_52w_high"]
    # Distance from SMA20 measured in ATR units (used for the "extended" risk flag).
    out["stretch_atr"] = (close - out["sma_20"]) / out[f"atr_{atr_period}"]
    # Slope of SMA50 over 10 bars as % of price, a simple trend-strength proxy.
    out["sma_50_slope_pct"] = (out["sma_50"] - out["sma_50"].shift(10)) / close * 100.0
    return out


@dataclass
class IndicatorSnapshot:
    """Latest indicator values for one symbol, suitable for JSON."""

    as_of: pd.Timestamp
    values: Dict[str, Optional[float]]

    def to_dict(self) -> dict:
        return {"as_of": self.as_of.strftime("%Y-%m-%d"), **self.values}


def latest_snapshot(ind: pd.DataFrame) -> IndicatorSnapshot:
    row = ind.iloc[-1]
    values = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    return IndicatorSnapshot(as_of=ind.index[-1], values=values)
