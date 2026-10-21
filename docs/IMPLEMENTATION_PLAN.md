# MarketIQ Implementation Plan

Date: 2026-10-21
Base repository: AI4Finance-Foundation/FinRL-Trading (FinRL-X), commit `4409abe`, Apache 2.0.

## 1. What is in the FinRL-X repository

23k lines of Python across `src/`. The repo is a portfolio-weight trading framework: a strategy emits a target weight vector, a `bt`-based engine backtests it, and an Alpaca adapter executes it. It is not a stock research or signal platform, so most of MarketIQ is new, but several pieces are worth keeping.

### Bundled real data (the most valuable asset)

`data/finrl_trading.7z` unpacks to a 423 MB SQLite database:

| Table | Rows | Coverage |
|---|---|---|
| `price_data` | 1,275,978 | Daily OHLCV + adj_close for 506 tickers, 2015-06-15 to 2025-10-16 (SPY/QQQ from 2020-12-01 to 2025-10-07) |
| `raw_payloads` | 77,364 | FMP quarterly income, balance sheet, cash flow and ratio payloads for 503 tickers, 2015-10 to 2025-09 |
| `sp500_components_details` | 2 | S&P 500 constituents snapshot with sectors |

Plus `data/fundamental_data_full.csv` (22,909 processed quarterly factor rows, 715 tickers, to 2026-03-31) and `data/sp500_historical_constituents.csv`.

This means MarketIQ runs end to end on real market data without any API key, and tests run against real prices. Live providers (yfinance, FMP) extend the dataset forward from 2025-10-16 when run on a machine with internet access. The sandbox used for this build has no market-data network access, so everything here is verified against the bundled data; the live fetch path is built against the same provider interface and exercised with recorded fixtures.

### Reuse / modify / build matrix

| FinRL-X component | Decision | How it is used in MarketIQ |
|---|---|---|
| `src/data/trading_calendar.py` | Reuse (copy, light cleanup) | NYSE trading-day calendar, missing-range detection for incremental fetches |
| `src/data/data_store.py` SQLite schema | Modify | Schema concepts (price_data, raw_payloads, news_articles, news_fetch_log) carried into SQLAlchemy models so Postgres is a URL change. Importer reads the FinRL-X DB directly |
| `src/data/data_fetcher.py` (FMP) | Modify | FMP endpoints, incremental-range logic and news cache pattern re-implemented as a `FMPProvider` behind a provider interface; yfinance added as the free default |
| `src/data/data_fetcher.py` GPT sentiment | Modify | Optional `OpenAISentimentAnalyzer`; default is an offline financial lexicon so the platform works without a key |
| `src/config/settings.py` | Modify | Same pydantic-settings pattern, new settings groups, `ENABLE_LIVE_TRADING` hard default false |
| `src/backtest/backtest_engine.py` | Partial reuse | Metric formulas (Sharpe, Sortino, Calmar, max drawdown, win rate, monthly stats). The `bt` weight-vector engine is not suited to per-stock signal backtests, so an event-driven signal backtester is new |
| `src/strategies/adaptive_rotation/walk_forward.py` | Reuse pattern | Point-in-time window generation pattern reused for ML walk-forward validation |
| `src/trading/alpaca_manager.py` | Reuse (optional) | Kept as the optional Alpaca paper broker adapter. Disabled unless configured; live endpoints blocked unless `ENABLE_LIVE_TRADING=true` |
| `src/strategies/ml_strategy.py` | Reference only | Quarterly fundamental RF selection. Not reused; MarketIQ's ML predicts a defined forward price direction from daily features |
| `src/web/app.py`, `src/tools/dashboard.py` (Streamlit) | Retire | Replaced by FastAPI + Next.js |
| `src/strategies/adaptive_rotation/*`, DRL, rotation configs | Retire | Out of scope; preserved under `legacy/` |
| `deploy.sh`, Dockerfile | Replace | New Dockerfile/compose for API + web |
| `LICENSE`, citation | Keep | Apache 2.0 retained; `NOTICE` added with FinRL-X attribution and adapted-file list |

Original code is moved to `legacy/finrlx/` untouched so attribution and history are preserved. Adapted files carry a header stating their origin.

## 2. Target architecture

```
marketiq/
  config/        settings (pydantic), universe.yaml, scoring.yaml, signals.yaml
  data/          SQLAlchemy models, repository, providers (local_db, yfinance, fmp), importer, trading calendar
  analysis/      indicators.py, patterns.py, fundamentals.py, sentiment.py
  scoring/       scorer.py (0-100, weighted components, explanations)
  signals/       engine.py (BUY/SELL/HOLD/WATCH rules, explanations)
  ml/            features.py, model.py (LightGBM/sklearn), walk_forward.py, registry.py
  backtest/      engine.py (event-driven), metrics.py (from FinRL-X), walk_forward.py
  paper/         broker.py (simulated), alpaca_adapter.py (optional)
  api/           FastAPI app, routers
  cli.py         marketiq init-db | import-finrlx | fetch | scan | train | backtest | serve
web/             Next.js 14 (app router, TypeScript, Tailwind, lightweight-charts)
tests/           pytest, fixtures sliced from the real bundled data
docs/            ARCHITECTURE.md, SIGNALS.md, SCORING.md, this plan
legacy/finrlx/   original FinRL-X source
```

Key decisions:

- **SQLite by default, Postgres by URL.** All persistence goes through SQLAlchemy 2.x models; `DATABASE_URL=postgresql+psycopg://...` switches engines. No raw SQL in application code.
- **Provider interface.** `PriceProvider`, `FundamentalsProvider`, `NewsProvider` protocols. `LocalDBProvider` serves the bundled data; `YFinanceProvider` is the free live default; `FMPProvider` when a key exists. The scanner never calls a provider directly; it reads the repository, and `fetch` fills gaps using the trading calendar.
- **Everything explainable.** Every indicator value, pattern flag, score component and signal carries a text reason. The API returns the full breakdown; the UI renders it.
- **No fabricated results.** Where a data source is absent (news in the sandbox) the component is marked `unavailable` and its weight is renormalized across available components. The score explanation says so.
- **Point-in-time discipline.** Indicators use only past data; ML labels use forward returns with a strict gap; walk-forward windows never see the future.
- **Real-money trading is off.** The paper broker is internal and default. Alpaca is optional and paper-only unless an explicit env flag plus non-paper URL are both set.

## 3. Milestones and acceptance criteria

### M1. Skeleton, config, database, data import
- `marketiq` package, `pyproject.toml`, `.env.example`, `config/*.yaml`.
- SQLAlchemy models: tickers, price_bars, fundamentals_raw, fundamentals_quarterly, news_articles, news_fetch_log, score_snapshots, signal_snapshots, watchlist, paper_accounts/positions/trades, backtest_runs, ml_models.
- `marketiq import-finrlx data/finrl_trading.db` loads the 20-30 ticker universe (plus SPY) from the bundled DB.
- `fetch` command uses provider interface with calendar-based gap detection.
- Accept: `marketiq init-db && marketiq import-finrlx` loads > 2,500 bars per universe ticker; `pytest tests/data` green; yfinance/FMP providers unit-tested with recorded fixtures.

### M2. Technical indicators and pattern detection
- RSI(14, Wilder), MACD(12,26,9), SMA 20/50/200, EMA 12/26, volume SMA20 and ratio, OBV, ATR(14), Bollinger(20,2), 52-week high/low distance, returns 1/5/20/60d.
- Patterns: trend regime (SMA stack and slope), golden/death cross, 20d and 52w breakout/breakdown with volume confirmation, momentum shift (MACD cross, RSI 50 cross), reversal candidates (RSI extremes, bullish/bearish divergence, stretch from SMA20 in ATR units).
- Accept: indicator tests match hand-computed values on a synthetic series to 1e-6 and match a reference computation on real AAPL data; pattern tests cover each detector on constructed series; `scan` prints indicators for the whole universe from local data.

### M3. Fundamentals, sentiment, scoring, signals
- Fundamentals from FMP payloads: revenue/EPS growth YoY, gross/operating/net margin, ROE, debt-to-equity, current ratio, FCF/share, P/E, P/B, P/S, PEG, dividend yield, last reporting date, days since report; universe percentile ranks.
- News: yfinance and FMP providers with DB cache; sentiment via offline lexicon (default) or OpenAI (optional). Aggregated per ticker over a window with decay.
- Scorer: components trend, momentum, volume, volatility/risk, fundamentals, sentiment, ml (optional). Each 0-100 with factor contributions. Weights in `config/scoring.yaml`, validated, renormalized over available components. Output includes `explanation[]`.
- Signals: rules documented in `docs/SIGNALS.md`; BUY, SELL, HOLD, WATCH with reasons and risk flags (earnings within N days, extended from SMA20, high ATR%).
- Accept: deterministic tests for scorer (weights sum, bounds, renormalization, explanation presence) and for every signal rule; `scan` ranks the universe and persists score + signal snapshots.

### M4. ML direction model
- Features from M2 outputs; label = forward return over `horizon_days` (default 20) above `threshold` (default 0); LightGBM classifier with sklearn fallback.
- Time-series split evaluation and walk-forward (expanding window, step = horizon) reporting accuracy, balanced accuracy, AUC, precision at top decile, per fold and aggregate. Nothing reported that was not computed.
- Model registry under `models/` with metadata JSON (features, horizon, train range, metrics).
- Accept: `marketiq train` completes on real local data and writes metrics; `predict` returns probability per ticker; tests verify no label leakage (feature date < label window start) and that the pipeline runs on a small fixture.

### M5. Backtesting and paper trading
- Event-driven daily backtester: takes signal rules + scoring config, long-only, position sizing by equal weight or score, max positions, commission and slippage, fills at next open; outputs equity curve, trades, metrics (FinRL-X formulas), benchmark comparison vs SPY.
- Walk-forward harness: rolling windows re-fit ML and evaluate signal strategy out of sample; aggregated report.
- Paper broker: cash, positions, market orders filled at next bar; persisted; P&L and history endpoints. Alpaca paper adapter optional. `ENABLE_LIVE_TRADING` defaults false and non-paper URLs are refused.
- Accept: backtest tests on constructed price paths verify fills, costs, drawdown and that no trade uses same-day information; paper broker tests cover buy/sell/insufficient cash; `marketiq backtest` runs on 2021-2025 local data and prints real computed metrics.

### M6. FastAPI backend
- Routers: health, universe, stocks (bars, indicators, patterns, fundamentals, news, score, signal), rankings, watchlist, ml, backtest, paper, config.
- Accept: `pytest tests/api` green using TestClient against a seeded SQLite fixture; OpenAPI docs served at `/docs`.

### M7. Next.js dashboard
- Pages: Rankings (sortable table, signal badges, score bars), Stock detail (candlestick with SMA overlays, RSI/MACD/volume panels, score breakdown, signal explanation, fundamentals card, news with sentiment), Watchlist, Backtest runner with equity chart and metrics, Paper trading, Settings (weights editor).
- Accept: `npm run build` passes with zero type errors; pages load against the running API; charts render real data.

### M8. Documentation, attribution, verification
- README (setup, env template, run, architecture), ARCHITECTURE.md, SIGNALS.md, SCORING.md, NOTICE.
- Accept: fresh-clone instructions reproduced in the sandbox; full `pytest` run recorded; one full scan + train + backtest cycle recorded with outputs; open items listed honestly.

## 4. Out of scope for v1
Options, intraday data, portfolio optimization, multi-user auth, real-money execution.
