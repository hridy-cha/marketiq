"""
Pattern and regime detection on top of the indicator frame.

Each detector returns a `PatternHit` with a boolean, a direction (bullish/bearish/neutral), the bar
on which it last fired, and a plain-English reason. `detect_all` evaluates every detector for the
latest bar and also reports which fired within the last `recent_bars` bars, which is what the signal
rules consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PatternHit:
    name: str
    fired: bool
    direction: str  # bullish | bearish | neutral
    last_fired: Optional[str]  # ISO date of the most recent bar where the condition was true
    bars_ago: Optional[int]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _last_true(mask: pd.Series) -> tuple[Optional[str], Optional[int]]:
    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return None, None
    idx = mask[mask].index[-1]
    bars_ago = int(len(mask) - 1 - mask.index.get_loc(idx))
    return idx.strftime("%Y-%m-%d"), bars_ago


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


# ---------------------------------------------------------------------------
# Trend regime
# ---------------------------------------------------------------------------
def trend_regime(ind: pd.DataFrame) -> PatternHit:
    """Classify the latest bar: strong_bullish, bullish, neutral, bearish, strong_bearish."""
    r = ind.iloc[-1]
    c, s20, s50, s200 = r["close"], r.get("sma_20"), r.get("sma_50"), r.get("sma_200")
    if any(pd.isna(v) for v in (s20, s50, s200)):
        return PatternHit("trend_regime", False, "neutral", None, None, "Not enough history for SMA 200")
    if c > s20 > s50 > s200:
        return PatternHit("trend_regime", True, "bullish", ind.index[-1].strftime("%Y-%m-%d"), 0,
                          f"Strong uptrend: close {c:.2f} > SMA20 {s20:.2f} > SMA50 {s50:.2f} > SMA200 {s200:.2f}")
    if c > s50 and s50 > s200:
        return PatternHit("trend_regime", True, "bullish", ind.index[-1].strftime("%Y-%m-%d"), 0,
                          f"Uptrend: close above SMA50 ({s50:.2f}) and SMA50 above SMA200 ({s200:.2f})")
    if c < s20 < s50 < s200:
        return PatternHit("trend_regime", True, "bearish", ind.index[-1].strftime("%Y-%m-%d"), 0,
                          f"Strong downtrend: close {c:.2f} < SMA20 {s20:.2f} < SMA50 {s50:.2f} < SMA200 {s200:.2f}")
    if c < s50 and s50 < s200:
        return PatternHit("trend_regime", True, "bearish", ind.index[-1].strftime("%Y-%m-%d"), 0,
                          f"Downtrend: close below SMA50 ({s50:.2f}) and SMA50 below SMA200 ({s200:.2f})")
    return PatternHit("trend_regime", True, "neutral", ind.index[-1].strftime("%Y-%m-%d"), 0,
                      f"Mixed trend: close {c:.2f}, SMA20 {s20:.2f}, SMA50 {s50:.2f}, SMA200 {s200:.2f}")


def golden_cross(ind: pd.DataFrame) -> PatternHit:
    mask = _cross_up(ind["sma_50"], ind["sma_200"])
    d, n = _last_true(mask)
    return PatternHit("golden_cross", n is not None, "bullish", d, n,
                      f"SMA50 crossed above SMA200 on {d}" if d else "No golden cross in history")


def death_cross(ind: pd.DataFrame) -> PatternHit:
    mask = _cross_down(ind["sma_50"], ind["sma_200"])
    d, n = _last_true(mask)
    return PatternHit("death_cross", n is not None, "bearish", d, n,
                      f"SMA50 crossed below SMA200 on {d}" if d else "No death cross in history")


# ---------------------------------------------------------------------------
# Breakouts
# ---------------------------------------------------------------------------
def breakout_20d(ind: pd.DataFrame, volume_confirm: float = 1.5) -> PatternHit:
    """Close above the prior 20-day high with volume >= volume_confirm x 20d average."""
    prior_high = ind["high_20"].shift(1)
    mask = (ind["close"] > prior_high) & (ind["volume_ratio"] >= volume_confirm)
    d, n = _last_true(mask)
    return PatternHit("breakout_20d", n is not None, "bullish", d, n,
                      f"Closed above prior 20-day high on {d} with volume {ind.loc[d, 'volume_ratio']:.1f}x average"
                      if d else "No volume-confirmed 20-day breakout")


def breakdown_20d(ind: pd.DataFrame, volume_confirm: float = 1.5) -> PatternHit:
    prior_low = ind["low_20"].shift(1)
    mask = (ind["close"] < prior_low) & (ind["volume_ratio"] >= volume_confirm)
    d, n = _last_true(mask)
    return PatternHit("breakdown_20d", n is not None, "bearish", d, n,
                      f"Closed below prior 20-day low on {d} with volume {ind.loc[d, 'volume_ratio']:.1f}x average"
                      if d else "No volume-confirmed 20-day breakdown")


def breakout_52w(ind: pd.DataFrame) -> PatternHit:
    prior_high = ind["high_52w"].shift(1)
    mask = ind["close"] > prior_high
    d, n = _last_true(mask)
    return PatternHit("breakout_52w", n is not None, "bullish", d, n,
                      f"New 52-week closing high on {d}" if d else "No new 52-week high")


# ---------------------------------------------------------------------------
# Momentum changes
# ---------------------------------------------------------------------------
def macd_bullish_cross(ind: pd.DataFrame) -> PatternHit:
    mask = _cross_up(ind["macd"], ind["macd_signal"])
    d, n = _last_true(mask)
    return PatternHit("macd_bullish_cross", n is not None, "bullish", d, n,
                      f"MACD crossed above its signal line on {d}" if d else "No MACD bullish cross")


def macd_bearish_cross(ind: pd.DataFrame) -> PatternHit:
    mask = _cross_down(ind["macd"], ind["macd_signal"])
    d, n = _last_true(mask)
    return PatternHit("macd_bearish_cross", n is not None, "bearish", d, n,
                      f"MACD crossed below its signal line on {d}" if d else "No MACD bearish cross")


def rsi_cross_50(ind: pd.DataFrame, rsi_col: str = "rsi_14") -> PatternHit:
    up = _cross_up(ind[rsi_col], pd.Series(50.0, index=ind.index))
    down = _cross_down(ind[rsi_col], pd.Series(50.0, index=ind.index))
    du, nu = _last_true(up)
    dd, nd = _last_true(down)
    if nu is not None and (nd is None or nu < nd):
        return PatternHit("rsi_cross_50", True, "bullish", du, nu, f"RSI crossed above 50 on {du}")
    if nd is not None:
        return PatternHit("rsi_cross_50", True, "bearish", dd, nd, f"RSI crossed below 50 on {dd}")
    return PatternHit("rsi_cross_50", False, "neutral", None, None, "RSI has not crossed 50")


# ---------------------------------------------------------------------------
# Reversal candidates
# ---------------------------------------------------------------------------
def rsi_oversold_reversal(ind: pd.DataFrame, rsi_col: str = "rsi_14", level: float = 30.0) -> PatternHit:
    """RSI was below `level` within the last 10 bars and has since turned up above it."""
    r = ind[rsi_col]
    was_oversold = (r.shift(1).rolling(10, min_periods=1).min() < level)
    mask = was_oversold & (r > level) & (r.shift(1) <= level)
    d, n = _last_true(mask)
    return PatternHit("rsi_oversold_reversal", n is not None, "bullish", d, n,
                      f"RSI recovered above {level:.0f} from oversold on {d}" if d else "No oversold reversal")


def rsi_overbought_reversal(ind: pd.DataFrame, rsi_col: str = "rsi_14", level: float = 70.0) -> PatternHit:
    r = ind[rsi_col]
    was_overbought = (r.shift(1).rolling(10, min_periods=1).max() > level)
    mask = was_overbought & (r < level) & (r.shift(1) >= level)
    d, n = _last_true(mask)
    return PatternHit("rsi_overbought_reversal", n is not None, "bearish", d, n,
                      f"RSI dropped below {level:.0f} from overbought on {d}" if d else "No overbought reversal")


def divergence(ind: pd.DataFrame, lookback: int = 40, rsi_col: str = "rsi_14") -> PatternHit:
    """Classic divergence over the lookback window: price makes a new extreme but RSI does not.

    Bullish: close at window low while RSI is above its value at the previous window low.
    Bearish: close at window high while RSI is below its value at the previous window high.
    Evaluated on the latest bar only."""
    if len(ind) < lookback + 5:
        return PatternHit("divergence", False, "neutral", None, None, "Not enough history for divergence check")
    win = ind.iloc[-lookback:]
    close, r = win["close"], win[rsi_col]
    last_date = ind.index[-1].strftime("%Y-%m-%d")
    # Bearish: today's close is the window high, compare with the prior high (excluding last 5 bars).
    if close.iloc[-1] >= close.max() and len(win) > 10:
        prior = win.iloc[:-5]
        prev_high_idx = prior["close"].idxmax()
        if r.iloc[-1] < prior.loc[prev_high_idx, rsi_col] - 3:
            return PatternHit("divergence", True, "bearish", last_date, 0,
                              f"Bearish divergence: price at {lookback}-day high but RSI {r.iloc[-1]:.0f} below "
                              f"{prior.loc[prev_high_idx, rsi_col]:.0f} at the prior high ({prev_high_idx.date()})")
    if close.iloc[-1] <= close.min() and len(win) > 10:
        prior = win.iloc[:-5]
        prev_low_idx = prior["close"].idxmin()
        if r.iloc[-1] > prior.loc[prev_low_idx, rsi_col] + 3:
            return PatternHit("divergence", True, "bullish", last_date, 0,
                              f"Bullish divergence: price at {lookback}-day low but RSI {r.iloc[-1]:.0f} above "
                              f"{prior.loc[prev_low_idx, rsi_col]:.0f} at the prior low ({prev_low_idx.date()})")
    return PatternHit("divergence", False, "neutral", None, None, "No RSI divergence at the latest bar")


def mean_reversion_stretch(ind: pd.DataFrame, atr_mult: float = 3.0) -> PatternHit:
    """Price more than `atr_mult` ATRs away from SMA20: reversal risk (up) or bounce candidate (down)."""
    s = ind["stretch_atr"].iloc[-1]
    last_date = ind.index[-1].strftime("%Y-%m-%d")
    if pd.isna(s):
        return PatternHit("stretch", False, "neutral", None, None, "Stretch unavailable")
    if s >= atr_mult:
        return PatternHit("stretch", True, "bearish", last_date, 0,
                          f"Extended: close is {s:.1f} ATR above SMA20 (reversal risk)")
    if s <= -atr_mult:
        return PatternHit("stretch", True, "bullish", last_date, 0,
                          f"Oversold stretch: close is {abs(s):.1f} ATR below SMA20 (bounce candidate)")
    return PatternHit("stretch", False, "neutral", None, None, f"Close is {s:.1f} ATR from SMA20 (normal)")


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
DETECTORS = [
    trend_regime,
    golden_cross,
    death_cross,
    breakout_20d,
    breakdown_20d,
    breakout_52w,
    macd_bullish_cross,
    macd_bearish_cross,
    rsi_cross_50,
    rsi_oversold_reversal,
    rsi_overbought_reversal,
    divergence,
    mean_reversion_stretch,
]


def detect_all(ind: pd.DataFrame, recent_bars: int = 5) -> Dict[str, dict]:
    """Run every detector. Adds `recent` = fired within the last `recent_bars` bars."""
    out: Dict[str, dict] = {}
    for fn in DETECTORS:
        try:
            hit = fn(ind)
        except Exception as exc:  # a detector must never break the scan
            hit = PatternHit(fn.__name__, False, "neutral", None, None, f"detector error: {exc}")
        d = hit.to_dict()
        d["recent"] = bool(hit.bars_ago is not None and hit.bars_ago <= recent_bars)
        out[hit.name] = d
    return out


def recent_pattern_names(patterns: Dict[str, dict], direction: Optional[str] = None) -> List[str]:
    return [k for k, v in patterns.items() if v.get("recent") and (direction is None or v["direction"] == direction)]
