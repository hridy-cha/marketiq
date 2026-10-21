"""
Signal engine: BUY / SELL / HOLD / WATCH from the composite score, detected patterns and risk flags.

Rules are read from config/signals.yaml and documented in docs/SIGNALS.md. Evaluation order:
1. Not enough history -> HOLD (insufficient data).
2. SELL if any configured sell condition is true.
3. BUY if score >= buy_score and all buy requirements hold and no blocking risk flag.
4. WATCH if score in [watch_low, buy_score) and at least one catalyst fired, or if a BUY was blocked
   only by a risk flag (so the user sees the near-miss).
5. HOLD otherwise.
Every result lists the rule that fired, the conditions that were true, and all active risk flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from marketiq.config.loaders import SignalConfig
from marketiq.scoring.scorer import ScoreResult

SIGNALS = ("BUY", "SELL", "HOLD", "WATCH")


@dataclass
class SignalResult:
    symbol: str
    as_of: str
    signal: str
    rule: str
    score: float
    reasons: List[str]
    risk_flags: List[str] = field(default_factory=list)
    conditions: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _nan(v) -> bool:
    return v is None or pd.isna(v)


def risk_flags(row: pd.Series, cfg: SignalConfig, next_earnings: Optional[date], as_of: date) -> List[str]:
    rf = cfg.risk_flags
    flags: List[str] = []
    if next_earnings is not None:
        days = (next_earnings - as_of).days
        if 0 <= days <= rf.earnings_within_days:
            flags.append(f"earnings in {days} day(s) on {next_earnings.isoformat()}")
    stretch = row.get("stretch_atr")
    if not _nan(stretch) and stretch >= rf.max_stretch_atr:
        flags.append(f"extended: {stretch:.1f} ATR above SMA20 (limit {rf.max_stretch_atr:.1f})")
    rsi = row.get("rsi_14")
    if not _nan(rsi):
        if rsi >= rf.rsi_overbought:
            flags.append(f"RSI overbought at {rsi:.0f}")
        elif rsi <= rf.rsi_oversold:
            flags.append(f"RSI oversold at {rsi:.0f}")
    atr_pct = row.get("atr_pct")
    if not _nan(atr_pct) and atr_pct >= rf.max_atr_pct:
        flags.append(f"high volatility: ATR {atr_pct:.1f}% of price")
    return flags


def evaluate(
    score: ScoreResult,
    patterns: Dict[str, dict],
    ind: pd.DataFrame,
    cfg: SignalConfig,
    next_earnings: Optional[date] = None,
    previous_score: Optional[float] = None,
) -> SignalResult:
    row = ind.iloc[-1]
    as_of = ind.index[-1].date()
    as_of_s = as_of.isoformat()
    th, rules = cfg.thresholds, cfg.rules
    flags = risk_flags(row, cfg, next_earnings, as_of)

    if len(ind) < th.min_history_days or _nan(row.get("sma_200")):
        return SignalResult(score.symbol, as_of_s, "HOLD", "insufficient_history", score.score,
                            [f"Only {len(ind)} bars of history; {th.min_history_days} required for a signal"], flags)

    def recent(name: str) -> bool:
        p = patterns.get(name)
        return bool(p and p.get("recent"))

    trend = patterns.get("trend_regime", {}).get("direction", "neutral")
    macd_hist = row.get("macd_hist")
    rsi = row.get("rsi_14")
    rsi_val = float(rsi) if not _nan(rsi) else 50.0

    conditions = {
        "score_at_or_above_buy": score.score >= th.buy_score,
        "score_at_or_below_sell": score.score <= th.sell_score,
        "trend_bullish": trend == "bullish",
        "trend_bearish": trend == "bearish",
        "death_cross_recent": recent("death_cross"),
        "golden_cross_recent": recent("golden_cross"),
        "breakdown_recent": recent("breakdown_20d"),
        "breakout_recent": recent("breakout_20d") or recent("breakout_52w"),
        "macd_bullish_cross_recent": recent("macd_bullish_cross"),
        "macd_bearish_cross_recent": recent("macd_bearish_cross"),
        "rsi_oversold_reversal": recent("rsi_oversold_reversal"),
        "trend_bearish_and_momentum_down": trend == "bearish" and not _nan(macd_hist) and macd_hist < 0 and rsi_val < 45,
        "score_rising": previous_score is not None and (score.score - previous_score) >= 8.0,
        "extended": any(f.startswith("extended") for f in flags),
        "rsi_overbought": any(f.startswith("RSI overbought") for f in flags),
        "earnings_soon": any(f.startswith("earnings") for f in flags),
    }

    # 2. SELL
    sell_hits = [c for c in rules.sell.any_of if conditions.get(c)]
    if sell_hits:
        reasons = [f"Sell condition met: {c.replace('_', ' ')}" for c in sell_hits]
        reasons.append(f"Composite score {score.score:.1f} (sell threshold {th.sell_score:.0f})")
        for c in sell_hits:
            key = {"death_cross_recent": "death_cross", "breakdown_recent": "breakdown_20d"}.get(c)
            if key and patterns.get(key):
                reasons.append(patterns[key]["reason"])
        return SignalResult(score.symbol, as_of_s, "SELL", "sell:" + "+".join(sell_hits), score.score, reasons, flags, conditions)

    # 3. BUY
    if conditions["score_at_or_above_buy"]:
        blockers: List[str] = []
        b = rules.buy
        if b.require_trend_bullish and not conditions["trend_bullish"]:
            blockers.append(f"trend is {trend}, bullish required")
        if b.require_no_sell_patterns and (conditions["death_cross_recent"] or conditions["breakdown_recent"]):
            blockers.append("a bearish pattern fired in the last 5 bars")
        if b.block_if_extended and conditions["extended"]:
            blockers.append("price is extended above SMA20")
        if b.block_if_rsi_overbought and conditions["rsi_overbought"]:
            blockers.append("RSI is overbought")
        if not blockers:
            reasons = [
                f"Composite score {score.score:.1f} is at or above the buy threshold {th.buy_score:.0f}",
                patterns.get("trend_regime", {}).get("reason", "trend bullish"),
            ]
            reasons += [patterns[k]["reason"] for k in ("breakout_20d", "breakout_52w", "golden_cross", "macd_bullish_cross") if recent(k)]
            if flags:
                reasons.append("Risk flags present: " + "; ".join(flags))
            return SignalResult(score.symbol, as_of_s, "BUY", "buy:score_and_trend", score.score, reasons, flags, conditions)
        reasons = [f"Score {score.score:.1f} qualifies for BUY but blocked: " + "; ".join(blockers)]
        return SignalResult(score.symbol, as_of_s, "WATCH", "watch:buy_blocked", score.score, reasons, flags, conditions)

    # 4. WATCH
    if th.watch_low <= score.score < th.buy_score:
        catalysts = [c for c in rules.watch.catalysts if conditions.get(c)]
        if catalysts:
            reasons = [f"Score {score.score:.1f} in watch band [{th.watch_low:.0f}, {th.buy_score:.0f}) with catalyst: "
                       + ", ".join(c.replace("_", " ") for c in catalysts)]
            for c in catalysts:
                key = {"breakout_recent": "breakout_20d", "golden_cross_recent": "golden_cross",
                       "macd_bullish_cross_recent": "macd_bullish_cross", "rsi_oversold_reversal": "rsi_oversold_reversal"}.get(c)
                if key and patterns.get(key, {}).get("recent"):
                    reasons.append(patterns[key]["reason"])
                elif key == "breakout_20d" and patterns.get("breakout_52w", {}).get("recent"):
                    reasons.append(patterns["breakout_52w"]["reason"])
            if conditions["score_rising"]:
                reasons.append(f"Score rose {score.score - previous_score:+.1f} points since the previous scan")
            return SignalResult(score.symbol, as_of_s, "WATCH", "watch:catalyst", score.score, reasons, flags, conditions)

    # 5. HOLD
    reasons = [f"Score {score.score:.1f} is between the sell ({th.sell_score:.0f}) and buy ({th.buy_score:.0f}) thresholds with no sell condition"]
    if th.watch_low <= score.score < th.buy_score:
        reasons.append("In the watch band but no catalyst fired recently")
    reasons.append(patterns.get("trend_regime", {}).get("reason", ""))
    return SignalResult(score.symbol, as_of_s, "HOLD", "hold:default", score.score, [r for r in reasons if r], flags, conditions)
