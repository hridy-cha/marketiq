"""
Composite 0-100 scoring with full explanation.

Each component (trend, momentum, volume, volatility, fundamentals, sentiment, ml) is scored from
named factors, each factor mapped to 0-100 by an explicit formula documented in docs/SCORING.md.
A component with no usable factors is marked unavailable and excluded; the remaining weights are
renormalized so the composite stays on the 0-100 scale. The result carries every factor value,
every factor score, the weights actually used and a list of plain-English explanation lines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from marketiq.config.loaders import ScoringConfig


@dataclass
class Factor:
    name: str
    value: Optional[float]
    score: Optional[float]
    note: str


@dataclass
class Component:
    name: str
    score: Optional[float]
    weight: float  # weight actually applied after renormalization (0 if unavailable)
    configured_weight: float
    available: bool
    factors: List[Factor] = field(default_factory=list)
    note: str = ""


@dataclass
class ScoreResult:
    symbol: str
    as_of: str
    score: float
    components: Dict[str, Component]
    explanation: List[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "score": self.score,
            "components": {k: asdict(v) for k, v in self.components.items()},
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (hasattr(v, "__float__") and pd.isna(v))


def linear(value: Optional[float], lo: float, hi: float) -> Optional[float]:
    """Map value linearly from [lo, hi] onto [0, 100] (clipped). lo > hi inverts the direction."""
    if _nan(value):
        return None
    if lo == hi:
        return 50.0
    t = (float(value) - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0) * 100.0)


def _mean(scores: List[Optional[float]]) -> Optional[float]:
    vals = [s for s in scores if s is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(v: Optional[float], pct: bool = False, digits: int = 2) -> str:
    if _nan(v):
        return "n/a"
    return f"{v * 100:.{digits}f}%" if pct else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# component scorers
# ---------------------------------------------------------------------------
def score_trend(row: pd.Series, cfg: ScoringConfig) -> Component:
    c = cfg.trend
    close, s_f, s_m, s_s = row.get("close"), row.get(f"sma_{c.sma_fast}"), row.get(f"sma_{c.sma_mid}"), row.get(f"sma_{c.sma_slow}")
    factors: List[Factor] = []
    if any(_nan(v) for v in (close, s_f, s_m, s_s)):
        return Component("trend", None, 0.0, 0.0, False, [], "Not enough history for moving averages")

    if close > s_f > s_m > s_s:
        align, note = 100.0, "price > SMA20 > SMA50 > SMA200 (full bullish stack)"
    elif close > s_m > s_s:
        align, note = 80.0, "price > SMA50 > SMA200"
    elif close > s_s and s_m > s_s:
        align, note = 65.0, "price above SMA200 with SMA50 above SMA200, short-term pullback"
    elif close > s_s:
        align, note = 55.0, "price above SMA200 but SMA50 below SMA200"
    elif s_m > s_s:
        align, note = 40.0, "price below SMA200 while SMA50 still above SMA200"
    elif close > s_f:
        align, note = 25.0, "downtrend with a short-term bounce above SMA20"
    elif close < s_f < s_m < s_s:
        align, note = 0.0, "price < SMA20 < SMA50 < SMA200 (full bearish stack)"
    else:
        align, note = 15.0, "price below all major moving averages"
    factors.append(Factor("sma_alignment", None, align, note))

    pct_vs_200 = (close / s_s - 1.0) * 100.0
    factors.append(Factor("price_vs_sma200_pct", pct_vs_200, linear(pct_vs_200, -20, 20), f"price is {pct_vs_200:+.1f}% vs SMA200"))
    slope = row.get("sma_50_slope_pct")
    factors.append(Factor("sma50_slope_pct", slope, linear(slope, -5, 5), f"SMA50 slope over 10 bars {_fmt(slope)}% of price"))
    hi, lo = row.get("high_52w"), row.get("low_52w")
    pos = None if (_nan(hi) or _nan(lo) or hi == lo) else (close - lo) / (hi - lo) * 100.0
    factors.append(Factor("position_in_52w_range", pos, pos, f"close sits at {_fmt(pos, digits=0)}% of the 52-week range"))
    score = _mean([f.score for f in factors])
    return Component("trend", score, cfg.weights.get("trend", 0), cfg.weights.get("trend", 0), score is not None, factors)


def score_momentum(row: pd.Series, cfg: ScoringConfig, bench_row: Optional[pd.Series]) -> Component:
    c = cfg.momentum
    factors: List[Factor] = []
    rsi = row.get(f"rsi_{c.rsi_period}")
    if not _nan(rsi):
        if c.rsi_ideal_low <= rsi <= c.rsi_ideal_high:
            dist = 0.0
        elif rsi < c.rsi_ideal_low:
            dist = c.rsi_ideal_low - rsi
        else:
            dist = rsi - c.rsi_ideal_high
        rsi_score = float(np.clip(100.0 - dist * (100.0 / 35.0), 0, 100))
        factors.append(Factor("rsi", rsi, rsi_score, f"RSI {rsi:.0f} (ideal band {c.rsi_ideal_low:.0f}-{c.rsi_ideal_high:.0f})"))
    hist, close = row.get("macd_hist"), row.get("close")
    if not _nan(hist) and not _nan(close) and close:
        hist_pct = hist / close * 100.0
        factors.append(Factor("macd_hist_pct", hist_pct, linear(hist_pct, -1.0, 1.0),
                              f"MACD histogram {'positive' if hist > 0 else 'negative'} ({hist_pct:+.2f}% of price)"))
    for w in c.return_windows:
        r = row.get(f"ret_{w}")
        if _nan(r):
            continue
        if bench_row is not None and not _nan(bench_row.get(f"ret_{w}")):
            rel = (r - bench_row.get(f"ret_{w}")) * 100.0
            span = 10.0 if w <= 20 else 20.0
            factors.append(Factor(f"rel_return_{w}d", rel, linear(rel, -span, span),
                                  f"{w}-day return {r * 100:+.1f}% vs benchmark {bench_row.get(f'ret_{w}') * 100:+.1f}%"))
        else:
            span = 10.0 if w <= 20 else 20.0
            factors.append(Factor(f"return_{w}d", r * 100, linear(r * 100, -span, span), f"{w}-day return {r * 100:+.1f}%"))
    score = _mean([f.score for f in factors])
    return Component("momentum", score, cfg.weights.get("momentum", 0), cfg.weights.get("momentum", 0), score is not None, factors)


def score_volume(row: pd.Series, cfg: ScoringConfig) -> Component:
    factors: List[Factor] = []
    obv, obv_sma, ret20, ret1, vr = row.get("obv"), row.get("obv_sma_20"), row.get("ret_20"), row.get("ret_1"), row.get("volume_ratio")
    if not any(_nan(v) for v in (obv, obv_sma, ret20)):
        accumulating, price_up = obv > obv_sma, ret20 > 0
        if accumulating and price_up:
            s, note = 100.0, "OBV above its 20-day average and price rising: volume confirms the move"
        elif accumulating:
            s, note = 65.0, "OBV above its 20-day average while price is flat or down: possible accumulation"
        elif price_up:
            s, note = 35.0, "price rising but OBV below its average: move lacks volume support"
        else:
            s, note = 0.0, "OBV below its average and price falling: distribution"
        factors.append(Factor("obv_confirmation", obv - obv_sma, s, note))
    if not _nan(vr) and not _nan(ret1):
        if vr >= 1.5:
            s = 100.0 if ret1 > 0 else 0.0
            note = f"volume {vr:.1f}x the prior 20-day average on a{'n up' if ret1 > 0 else ' down'} day"
        elif vr <= 0.6:
            s, note = 45.0, f"quiet session, volume {vr:.1f}x average"
        else:
            s, note = 60.0 if ret1 > 0 else 40.0, f"volume {vr:.1f}x average on a{'n up' if ret1 > 0 else ' down'} day"
        factors.append(Factor("volume_ratio", vr, s, note))
    score = _mean([f.score for f in factors])
    return Component("volume", score, cfg.weights.get("volume", 0), cfg.weights.get("volume", 0), score is not None, factors)


def score_volatility(row: pd.Series, cfg: ScoringConfig) -> Component:
    c = cfg.volatility
    factors: List[Factor] = []
    atr_pct = row.get("atr_pct")
    factors.append(Factor("atr_pct", atr_pct, linear(atr_pct, c.atr_pct_bad, c.atr_pct_good), f"ATR is {_fmt(atr_pct)}% of price"))
    rv = row.get("realized_vol_20")
    if not _nan(rv):
        factors.append(Factor("realized_vol_20", rv * 100, linear(rv * 100, 60, 15), f"20-day realized volatility {rv * 100:.0f}% annualized"))
    dd = row.get("drawdown_from_high_52w")
    factors.append(Factor("drawdown_from_52w_high", dd, linear(dd, -30, 0), f"{_fmt(dd, digits=1)}% below the 52-week high"))
    factors = [f for f in factors if f.score is not None]
    score = _mean([f.score for f in factors])
    return Component("volatility", score, cfg.weights.get("volatility", 0), cfg.weights.get("volatility", 0), score is not None, factors)


def score_fundamentals(factors_pct: Optional[Dict[str, Optional[float]]], raw: Optional[Dict[str, Optional[float]]],
                       cfg: ScoringConfig, min_factors: int = 3) -> Component:
    w = cfg.weights.get("fundamentals", 0)
    if not factors_pct:
        return Component("fundamentals", None, 0.0, w, False, [], "No fundamentals available")
    factors: List[Factor] = []
    for name, pct in factors_pct.items():
        if pct is None:
            continue
        value = (raw or {}).get(name)
        direction = cfg.fundamentals.factors.get(name, 1)
        factors.append(Factor(name, value, pct, f"{name.replace('_', ' ')} {_fmt(value, pct=name.endswith(('growth_yoy', 'margin', 'roe', 'yield')))}: "
                                                 f"{pct:.0f}th percentile in universe ({'higher' if direction > 0 else 'lower'} is better)"))
    if len(factors) < min_factors:
        return Component("fundamentals", None, 0.0, w, False, factors, f"Only {len(factors)} fundamental factor(s); need {min_factors}")
    score = _mean([f.score for f in factors])
    return Component("fundamentals", score, w, w, True, factors)


def score_sentiment(agg, cfg: ScoringConfig) -> Component:
    w = cfg.weights.get("sentiment", 0)
    if agg is None or not agg.available:
        return Component("sentiment", None, 0.0, w, False, [], (agg.reason if agg else "No news data"))
    s = (agg.score + 1.0) / 2.0 * 100.0
    f = Factor("news_sentiment", agg.score, s, agg.reason)
    return Component("sentiment", s, w, w, True, [f])


def score_ml(prob_up: Optional[float], cfg: ScoringConfig, horizon_days: Optional[int] = None,
             note: Optional[str] = None) -> Component:
    w = cfg.weights.get("ml", 0)
    if prob_up is None:
        return Component("ml", None, 0.0, w, False, [], note or "No trained model prediction")
    s = float(np.clip(prob_up, 0, 1)) * 100.0
    note = f"model probability of a positive return over {horizon_days or '?'} trading days: {prob_up:.2f}"
    return Component("ml", s, w, w, True, [Factor("prob_up", prob_up, s, note)])


# ---------------------------------------------------------------------------
# composite
# ---------------------------------------------------------------------------
def combine(symbol: str, as_of: str, components: Dict[str, Component]) -> ScoreResult:
    available = {k: c for k, c in components.items() if c.available and c.score is not None}
    total_w = sum(c.configured_weight for c in available.values())
    if total_w <= 0 or not available:
        for c in components.values():
            c.weight = 0.0
        return ScoreResult(symbol, as_of, 0.0, components, ["No scoring components available"])
    score = 0.0
    for k, c in components.items():
        if k in available:
            c.weight = c.configured_weight / total_w
            score += c.weight * c.score
        else:
            c.weight = 0.0
    score = round(float(np.clip(score, 0, 100)), 1)
    lines = [f"Composite score {score:.1f}/100 from {len(available)} of {len(components)} components"]
    ordered = sorted(available.values(), key=lambda c: c.weight * c.score, reverse=True)
    for c in ordered:
        lines.append(f"{c.name.capitalize()} {c.score:.0f}/100 (weight {c.weight:.0%}): " + "; ".join(f.note for f in c.factors[:3]))
    for c in components.values():
        if not c.available:
            lines.append(f"{c.name.capitalize()} unavailable: {c.note or 'no data'} (weight redistributed)")
    return ScoreResult(symbol, as_of, score, components, lines)


def score_symbol(
    symbol: str,
    ind: pd.DataFrame,
    cfg: ScoringConfig,
    bench_ind: Optional[pd.DataFrame] = None,
    fundamentals_pct: Optional[Dict[str, Optional[float]]] = None,
    fundamentals_raw: Optional[Dict[str, Optional[float]]] = None,
    sentiment_agg=None,
    ml_prob: Optional[float] = None,
    ml_horizon: Optional[int] = None,
    ml_note: Optional[str] = None,
) -> ScoreResult:
    row = ind.iloc[-1]
    as_of = ind.index[-1].strftime("%Y-%m-%d")
    bench_row = None
    if bench_ind is not None and not bench_ind.empty:
        # Align the benchmark to the same date (or the last bar at or before it).
        b = bench_ind.loc[:ind.index[-1]]
        bench_row = b.iloc[-1] if not b.empty else None
    components = {
        "trend": score_trend(row, cfg),
        "momentum": score_momentum(row, cfg, bench_row),
        "volume": score_volume(row, cfg),
        "volatility": score_volatility(row, cfg),
        "fundamentals": score_fundamentals(fundamentals_pct, fundamentals_raw, cfg),
        "sentiment": score_sentiment(sentiment_agg, cfg),
        "ml": score_ml(ml_prob, cfg, ml_horizon, ml_note),
    }
    # Drop components whose configured weight is zero so they neither count nor clutter the output.
    components = {k: v for k, v in components.items() if cfg.weights.get(k, 0) > 0}
    return combine(symbol, as_of, components)
