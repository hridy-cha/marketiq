# Architecture

MarketIQ is a research pipeline with a thin API and a dashboard on top. Data flows one way:

```
providers (local FinRL-X DB | yfinance | FMP)
   └─> repository (SQLAlchemy, SQLite/Postgres)
         └─> ScanContext: prices → indicators → patterns
                          fundamentals (point-in-time) · news sentiment · ML probability
                          └─> scorer (0-100 + explanation) ─> signal engine (BUY/SELL/HOLD/WATCH)
                                     └─> snapshots (score_snapshots, signal_snapshots)
                                           ├─> FastAPI ─> Next.js dashboard
                                           ├─> SignalBacktester / walk-forward
                                           └─> PaperBroker
```

## Packages

| Package | Responsibility | Key decisions |
|---|---|---|
| `marketiq.config` | pydantic settings from `.env`; typed loaders for `config/*.yaml` | Scoring weights validated and normalized at load; unknown components rejected |
| `marketiq.data` | ORM models, repository, trading calendar, providers, importer, ingestion | All persistence via SQLAlchemy so `DATABASE_URL` switches SQLite ↔ Postgres. Providers implement `PriceProvider` / `FundamentalsProvider` / `NewsProvider` protocols. Price gaps are computed against the NYSE calendar so live fetches only request missing ranges. Fundamentals are stored as raw statement rows plus filing dates |
| `marketiq.analysis` | indicators, pattern detectors, fundamentals factors, sentiment | Pure pandas, no look-ahead (tested). Fundamentals computed from statements (TTM) with provider ratios only as fallback; valuation uses the as-of price |
| `marketiq.scoring` | component scoring and composite with explanation | Every factor carries value, score and a sentence; unavailable components renormalize |
| `marketiq.signals` | rule engine from `config/signals.yaml` | Ordered rules, first match wins, near-miss BUYs surface as WATCH |
| `marketiq.scanner` | `ScanContext` loads once and evaluates as of any date | Shared by the CLI scan, API, backtester and walk-forward so all use identical logic |
| `marketiq.ml` | features, LightGBM/sklearn direction model, walk-forward validation, registry with validation gate | Labels are forward returns; folds leave a `horizon` gap after the training cut; a model only contributes to scores if its pooled walk-forward metrics clear the gate |
| `marketiq.backtest` | event-driven signal backtester, FinRL-X metric formulas, walk-forward harness | Decisions at the close, fills at the next open, costs in bps; walk-forward retrains the model per window and reports the gate outcome |
| `marketiq.paper` | simulated broker persisted to the DB; optional Alpaca adapter | No credentials needed; Alpaca refuses non-paper URLs unless `ENABLE_LIVE_TRADING=true` |
| `marketiq.api` | FastAPI routers | Rankings read persisted snapshots; per-stock detail recomputes from stored bars |
| `web/` | Next.js 14 app router, TypeScript, Tailwind, lightweight-charts, recharts, SWR | `/api/*` is rewritten to the FastAPI backend, so the browser talks to one origin |

## Point-in-time discipline
- Indicators at date t use bars ≤ t (`tests/analysis/test_indicators.py::test_compute_indicators_no_lookahead`).
- Fundamentals visible at t require filing date ≤ t, or period end + 45 days when the filing date is unknown (`repository.payload_visible_on`).
- ML labels use bars > t; walk-forward folds leave a `horizon` gap (`tests/ml/test_ml.py`).
- Backtest orders decided at the close of t fill at the open of t+1 (`tests/backtest/test_backtest.py`).

## Data
The bundled FinRL-X SQLite database (`data/finrl_trading.7z`, 423 MB extracted) holds daily OHLCV
for 506 tickers from 2015-06-15 to 2025-10-15 (the 2025-10-16 row is a partial intraday snapshot and
is dropped on import) plus FMP quarterly income, balance sheet, cash flow and ratio payloads for 503
tickers. SPY/QQQ start 2020-12-01. There is no news in the bundle. `marketiq fetch` extends prices,
statements, earnings dates and news from yfinance or FMP on a machine with internet access.

## Known limitations
- The universe is a hand-picked list of 29 current large caps, so backtests carry survivorship bias.
- The benchmark-relative momentum and ML features need SPY, which the bundle only has from 2020-12.
- The offline sentiment lexicon is small and coarse; use the OpenAI analyzer for anything serious.
- Earnings dates are not in the bundle; the earnings risk flag only works after a live fetch.
- Backtests are long-only, daily, with simple equal-slot sizing; no portfolio optimization.
- Scan evaluation is ~0.4 s per date for 29 stocks; a 4-year weekly backtest takes about 1.5 minutes.
