# Signal rules

Signals are BUY, SELL, HOLD or WATCH. Rules and thresholds live in `config/signals.yaml`;
implementation in `marketiq/signals/engine.py`. Every signal records the rule that fired, the
conditions that were true and the active risk flags, and the UI shows them verbatim.

Evaluation order (first match wins):

1. **Insufficient history → HOLD.** Fewer than `min_history_days` (210) bars, or no SMA200.
2. **SELL** if any of (`rules.sell.any_of`):
   - `score_at_or_below_sell`: composite ≤ 35
   - `death_cross_recent`: SMA50 crossed below SMA200 within the last 5 bars
   - `breakdown_recent`: close below the prior 20-day low with volume ≥ 1.5× the prior 20-day average, within 5 bars
   - `trend_bearish_and_momentum_down`: bearish trend regime and MACD histogram < 0 and RSI < 45
3. **BUY** if composite ≥ 70 and all of (`rules.buy`):
   - trend regime is bullish (close > SMA50 > SMA200)
   - no death cross or breakdown in the last 5 bars
   - not extended (close ≤ 3 ATR above SMA20)
   - RSI < 80
   If the score qualifies but a requirement blocks it, the result is **WATCH** (`watch:buy_blocked`)
   with the blocker named, so near-misses are visible.
4. **WATCH** if composite in [55, 70) and at least one catalyst (`rules.watch.catalysts`) fired in the
   last 5 bars: 20-day or 52-week breakout, golden cross, MACD bullish cross, RSI oversold reversal,
   or the score rose ≥ 8 points since the previous scan.
5. **HOLD** otherwise.

## Risk flags (informational, attached to every signal)
- earnings within 7 days (when an earnings date is stored)
- extended: close more than 3 ATR above SMA20
- RSI ≥ 80 (overbought) or ≤ 20 (oversold)
- high volatility: ATR ≥ 6% of price

## Pattern detectors (`marketiq/analysis/patterns.py`)
trend_regime, golden_cross, death_cross, breakout_20d, breakdown_20d, breakout_52w,
macd_bullish_cross, macd_bearish_cross, rsi_cross_50, rsi_oversold_reversal, rsi_overbought_reversal,
divergence (price at a 40-day extreme while RSI disagrees by > 3 points), stretch (±3 ATR from SMA20).
Each returns direction, the date it last fired, bars ago, and a sentence explaining it; `recent` means
within the last 5 bars.

## How the backtester and paper broker use signals
Enter on BUY (optionally WATCH), highest score first, up to `max_positions`, equal notional per slot.
Exit on SELL, score below `exit_score` (45), stop loss (15% below entry, checked on the close), or
`max_hold_days`. Orders decided at the close fill at the next open with slippage and commission.
