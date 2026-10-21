# Scoring

The composite score is a weighted mean of component scores, each 0–100. Weights live in
`config/scoring.yaml` and are normalized to sum to 1. If a component has no usable data for a ticker
it is marked `unavailable`, dropped, and the remaining weights are renormalized for that ticker; the
explanation lists which components were used. Implementation: `marketiq/scoring/scorer.py`.

Each factor below is mapped to 0–100. `linear(x, lo, hi)` means a straight line from 0 at `lo` to 100
at `hi`, clipped; `lo > hi` inverts the direction. Component score = mean of its available factors.

## Trend (default weight 0.22)
| Factor | Score |
|---|---|
| sma_alignment | close > SMA20 > SMA50 > SMA200: 100 · close > SMA50 > SMA200: 80 · close > SMA200 and SMA50 > SMA200: 65 · close > SMA200 only: 55 · SMA50 > SMA200 but close below SMA200: 40 · downtrend with close > SMA20: 25 · full bearish stack: 0 · else 15 |
| price_vs_sma200_pct | linear(% above SMA200, -20, +20) |
| sma50_slope_pct | linear(10-bar SMA50 change as % of price, -5, +5) |
| position_in_52w_range | (close − 52w low) / (52w high − 52w low) × 100 |

## Momentum (0.18)
| Factor | Score |
|---|---|
| rsi | 100 inside the ideal band [50, 70]; falls by 100/35 per RSI point outside it |
| macd_hist_pct | linear(MACD histogram as % of price, -1, +1) |
| rel_return_20d / 60d | linear(return minus benchmark return in %, ±10 for 20d, ±20 for 60d); plain return if no benchmark |

## Volume (0.10)
| Factor | Score |
|---|---|
| obv_confirmation | OBV above its 20-day average and 20d return > 0: 100 · OBV above, price flat/down: 65 · price up, OBV below: 35 · both down: 0 |
| volume_ratio | today's volume ÷ prior 20-day average: ≥1.5× on an up day 100, on a down day 0; ≤0.6× 45; otherwise 60 (up) / 40 (down) |

## Volatility (0.10), lower risk scores higher
| Factor | Score |
|---|---|
| atr_pct | linear(ATR14 as % of price, 5.0, 1.5) |
| realized_vol_20 | linear(annualized 20-day vol %, 60, 15) |
| drawdown_from_52w_high | linear(% below 52-week high, -30, 0) |

## Fundamentals (0.25)
Point-in-time: only statements filed on or before the as-of date count (rows without a filing date
become visible 45 days after period end). TTM = sum of the last four quarters. Factors
(`marketiq/analysis/fundamentals.py`): revenue growth YoY, EPS growth YoY, net margin, ROE, debt to
equity, current ratio, FCF per share, P/E (as-of price ÷ TTM diluted EPS), price to book. Each is
ranked across the universe as a percentile (0–100), with direction per `config/scoring.yaml`
(`-1` = lower is better, e.g. P/E). Component = mean percentile; needs at least 3 factors.

## Sentiment (0.10)
Recency-weighted mean of article sentiment scores (−1..1) over `lookback_days` with half-life
`half_life_days`, mapped to 0–100. Needs at least `min_articles` scored articles. Sentiment comes
from the offline lexicon (`LexiconSentimentAnalyzer`, a small finance word list with negation) or
OpenAI when `OPENAI_API_KEY` is set. Without stored news the component is unavailable.

## ML (0.05)
Model probability of a positive forward return over the horizon, × 100. Only used when the latest
model passes the validation gate (`ml.min_walk_forward_auc`, `ml.min_walk_forward_balanced_accuracy`);
otherwise unavailable with the gate message in the explanation.

## Worked example (NVDA as of 2025-10-15, bundled data)
Composite 65.5 from 5 of 7 components: trend 82 (w 26%), fundamentals 70 (w 29%), momentum 59
(w 21%), volatility 57 (w 12%), volume 38 (w 12%); sentiment unavailable (0 articles), ML
unavailable (gate failed). Weights shown are after renormalization over the five available components.
