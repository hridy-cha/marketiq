"""MarketIQ command-line interface."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from marketiq.config import get_settings, load_universe

app = typer.Typer(help="MarketIQ: explainable stock research platform", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    _setup_logging(verbose)


@app.command("init-db")
def init_db_cmd():
    """Create tables and register the universe tickers."""
    from marketiq.data import repository as repo
    from marketiq.data.db import init_db, session_scope

    init_db()
    universe = load_universe()
    with session_scope() as s:
        rows = [{"symbol": t.symbol, "name": t.name, "sector": t.sector} for t in universe.tickers]
        rows.append({"symbol": universe.benchmark, "name": "Benchmark", "sector": "", "is_benchmark": True})
        n = repo.upsert_tickers(s, rows)
    console.print(f"Database ready at {get_settings().database.resolved_url()} with {n} tickers.")


@app.command("import-finrlx")
def import_finrlx_cmd(
    db_path: Optional[Path] = typer.Option(None, help="Path to FinRL-X finrl_trading.db (default from settings)"),
):
    """Load prices and fundamentals for the universe from the bundled FinRL-X SQLite database."""
    from marketiq.data.db import init_db, session_scope
    from marketiq.data.importer import import_all

    init_db()
    src = db_path or get_settings().data.path("finrlx_db")
    universe = load_universe()
    with session_scope() as s:
        result = import_all(s, src, universe.all_symbols)
    table = Table(title="FinRL-X import")
    table.add_column("Symbol")
    table.add_column("Price bars", justify="right")
    table.add_column("Fundamental rows", justify="right")
    for sym in universe.all_symbols:
        table.add_row(sym, str(result["prices"].get(sym, 0)), str(result["fundamentals"].get(sym, 0)))
    console.print(table)


@app.command("fetch")
def fetch_cmd(
    start: str = typer.Option("2015-01-01", help="Earliest date to backfill"),
    prices: bool = typer.Option(True, help="Fetch missing price bars"),
    fundamentals: bool = typer.Option(False, help="Refresh quarterly statements and earnings dates"),
    news: bool = typer.Option(False, help="Fetch recent news and score sentiment"),
    provider: Optional[str] = typer.Option(None, help="yfinance or fmp (default from settings)"),
):
    """Fill gaps from the live provider. Needs internet access."""
    from marketiq.data import ingest
    from marketiq.data.db import init_db, session_scope
    from marketiq.data.providers.base import provider_for

    init_db()
    universe = load_universe()
    symbols = universe.all_symbols
    with session_scope() as s:
        if prices:
            p = provider_for("price", provider)
            res = ingest.fetch_missing_prices(s, p, symbols, date.fromisoformat(start))
            console.print(f"Prices ({p.name}): stored " + ", ".join(f"{k}={v}" for k, v in res.items()))
        if fundamentals:
            p = provider_for("fundamentals", provider)
            res = ingest.refresh_fundamentals(s, p, symbols)
            console.print(f"Fundamentals ({p.name}): " + ", ".join(f"{k}={v}" for k, v in res.items()))
        if news:
            from marketiq.analysis.sentiment import default_analyzer

            p = provider_for("news", provider)
            res = ingest.refresh_news(s, p, universe.symbols, analyzer=default_analyzer())
            console.print(f"News ({p.name}): new articles " + ", ".join(f"{k}={v}" for k, v in res.items()))


@app.command("scan")
def scan_cmd(
    as_of: Optional[str] = typer.Option(None, help="Evaluate as of this date (default: latest stored bar)"),
    no_persist: bool = typer.Option(False, help="Do not store score/signal snapshots"),
    no_ml: bool = typer.Option(False, help="Skip the ML component even if a model exists"),
    explain: Optional[str] = typer.Option(None, help="Print the full explanation for one symbol"),
):
    """Score and rank the universe, emit signals, and store snapshots."""
    from marketiq.data.db import init_db, session_scope
    from marketiq.ml.registry import load_predictor
    from marketiq.scanner import run_scan

    init_db()
    predictor, horizon, ml_status = (None, None, "ML skipped (--no-ml)") if no_ml else load_predictor()
    console.print(f"ML component: {ml_status}")
    with session_scope() as s:
        rows = run_scan(s, date.fromisoformat(as_of) if as_of else None, persist=not no_persist,
                        ml_predictor=predictor, ml_horizon=horizon, ml_status=ml_status)
    table = Table(title=f"MarketIQ scan as of {rows[0].as_of if rows else 'n/a'}")
    for col in ("#", "Symbol", "Score", "Signal", "Rule", "Trend", "RSI", "Close", "Risk flags"):
        table.add_column(col, justify="right" if col in ("Score", "RSI", "Close", "#") else "left")
    for i, r in enumerate(rows, 1):
        table.add_row(str(i), r.symbol, f"{r.score.score:.1f}", r.signal.signal, r.signal.rule,
                      r.patterns["trend_regime"]["direction"], f"{(r.indicators.get('rsi_14') or 0):.0f}",
                      f"{r.indicators.get('close'):.2f}", "; ".join(r.signal.risk_flags)[:60])
    console.print(table)
    if explain:
        for r in rows:
            if r.symbol == explain.upper():
                console.rule(f"{r.symbol} explanation")
                for line in r.score.explanation:
                    console.print("  " + line)
                console.print(f"Signal {r.signal.signal} ({r.signal.rule}):")
                for line in r.signal.reasons:
                    console.print("  - " + line)


@app.command("train")
def train_cmd(
    horizon: Optional[int] = typer.Option(None, help="Forward horizon in trading days (default ML_HORIZON_DAYS)"),
    threshold: Optional[float] = typer.Option(None, help="Return threshold for the positive class (default 0)"),
    start: Optional[str] = typer.Option(None, help="Only use bars from this date"),
    folds: int = typer.Option(5, help="Walk-forward folds (6-month test windows each)"),
):
    """Train the direction model with walk-forward validation and register it."""
    from marketiq.data.db import init_db, session_scope
    from marketiq.ml.registry import train_and_register

    init_db()
    with session_scope() as s:
        meta = train_and_register(s, horizon_days=horizon, threshold=threshold, start=start, n_folds=folds)
    wf = meta["walk_forward"]
    table = Table(title=f"Walk-forward validation: {meta['name']} {meta['version']} ({meta['backend']}, horizon {meta['horizon_days']}d)")
    for col in ("Fold", "Train end", "Test window", "n test", "Accuracy", "Bal. acc", "AUC", "Top-decile prec", "Top-decile fwd ret", "All fwd ret", "Base rate"):
        table.add_column(col, justify="right" if col not in ("Train end", "Test window") else "left")
    for f in wf["folds"]:
        table.add_row(str(f["fold"]), f["train_end"], f"{f['test_start']}..{f['test_end']}", str(f["n_test"]),
                      f"{f['accuracy']:.3f}", f"{f['balanced_accuracy']:.3f}", f"{f['auc']:.3f}" if f["auc"] is not None else "n/a",
                      f"{f['top_decile_precision']:.3f}", f"{f['top_decile_mean_fwd_return'] * 100:+.2f}%",
                      f"{f['all_mean_fwd_return'] * 100:+.2f}%", f"{f['base_rate']:.3f}")
    p = wf["pooled"]
    if p:
        table.add_row("pooled", "", "", str(p["n_test_rows"]), f"{p['accuracy']:.3f}", f"{p['balanced_accuracy']:.3f}",
                      f"{p['auc']:.3f}" if p["auc"] is not None else "n/a", f"{p['top_decile_precision']:.3f}",
                      f"{p['top_decile_mean_fwd_return'] * 100:+.2f}%", f"{p['all_mean_fwd_return'] * 100:+.2f}%", f"{p['base_rate']:.3f}")
    console.print(table)
    console.print("Top features: " + ", ".join(f"{k} {v:.2f}" for k, v in list(wf["feature_importance"].items())[:8]))
    console.print(f"Saved to {meta['path']}")


def _print_metrics(title: str, metrics: dict):
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    pct = {"total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate", "benchmark_total_return",
           "benchmark_annual_return", "benchmark_max_drawdown", "excess_return", "trade_win_rate", "avg_exposure"}
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v * 100:.2f}%" if k in pct else f"{v:.3f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


@app.command("backtest")
def backtest_cmd(
    start: str = typer.Option("2021-01-04"),
    end: str = typer.Option("2025-10-15"),
    capital: float = typer.Option(100_000.0),
    max_positions: int = typer.Option(8),
    rebalance_every: int = typer.Option(5, help="Trading days between evaluations"),
    exit_score: float = typer.Option(45.0),
    stop_loss: Optional[float] = typer.Option(0.15, help="Fractional stop loss from entry; 0 disables"),
    enter_on_watch: bool = typer.Option(False, help="Also enter on WATCH signals"),
    no_ml: bool = typer.Option(False),
    save: bool = typer.Option(True, help="Persist the run to the database"),
    name: str = typer.Option("signal-strategy"),
):
    """Backtest the signal strategy on stored prices (fills at next open, costs included)."""
    from marketiq.backtest.engine import BacktestConfig, SignalBacktester
    from marketiq.data.db import init_db, session_scope
    from marketiq.data.models import BacktestRun
    from marketiq.ml.registry import load_predictor
    from marketiq.scanner import ScanContext

    init_db()
    predictor, horizon, status = (None, None, "ML skipped") if no_ml else load_predictor()
    cfg = BacktestConfig(start=date.fromisoformat(start), end=date.fromisoformat(end), initial_capital=capital,
                         max_positions=max_positions, rebalance_every=rebalance_every, exit_score=exit_score,
                         stop_loss_pct=stop_loss or None, enter_on=("BUY", "WATCH") if enter_on_watch else ("BUY",))
    with session_scope() as s:
        ctx = ScanContext.load(s, end=cfg.end, with_news=False, ml_predictor=predictor, ml_horizon=horizon, ml_status=status)
        console.print(f"ML component: {status}")
        res = SignalBacktester(ctx, cfg).run()
        if save:
            run = BacktestRun(name=name, kind="signal", start_date=cfg.start, end_date=cfg.end, config=cfg.to_dict(),
                              metrics=res.metrics, equity_curve=res.to_dict()["equity_curve"],
                              trades=[t.__dict__ for t in res.trades])
            s.add(run)
            s.commit()
            console.print(f"Saved as backtest run #{run.id}")
    _print_metrics(f"Backtest {name} {start}..{end} ({len(res.trades)} fills)", res.metrics)
    for t in res.trades[-10:]:
        console.print(f"  {t.date} {t.side:4} {t.symbol:6} {t.quantity:9.3f} @ {t.price:8.2f}  {t.reason}"
                      + (f"  pnl {t.pnl:+.2f}" if t.pnl is not None else ""))


@app.command("walk-forward")
def walk_forward_cmd(
    start: str = typer.Option("2022-01-03"),
    end: str = typer.Option("2025-10-15"),
    window_months: int = typer.Option(6),
    retrain_ml: bool = typer.Option(True),
    save: bool = typer.Option(True),
):
    """Walk-forward validation: retrain the ML model before each window, backtest inside it."""
    from marketiq.backtest.walk_forward import run_walk_forward
    from marketiq.data.db import init_db, session_scope
    from marketiq.data.models import BacktestRun

    init_db()
    with session_scope() as s:
        res = run_walk_forward(s, date.fromisoformat(start), date.fromisoformat(end), window_months, retrain_ml=retrain_ml)
        if save and len(res.stitched_equity):
            run = BacktestRun(name="walk-forward", kind="walk_forward", start_date=date.fromisoformat(start),
                              end_date=date.fromisoformat(end), config=res.config, metrics=res.aggregate,
                              equity_curve=res.to_dict()["equity_curve"], trades=[])
            s.add(run)
            s.commit()
            console.print(f"Saved as backtest run #{run.id}")
    table = Table(title="Walk-forward windows")
    for col in ("#", "Test window", "ML", "Return", "Max DD", "Sharpe", "Trades"):
        table.add_column(col, justify="right" if col not in ("Test window", "ML") else "left")
    for w in res.windows:
        m = w.metrics
        table.add_row(str(w.window), f"{w.test_start}..{w.test_end}", "yes" if w.ml_used else "no",
                      f"{m.get('total_return', 0) * 100:+.2f}%", f"{m.get('max_drawdown', 0) * 100:.2f}%",
                      f"{m.get('sharpe_ratio', 0):.2f}", str(w.n_trades))
    console.print(table)
    _print_metrics("Walk-forward aggregate (stitched)", res.aggregate)


paper_app = typer.Typer(help="Simulated paper trading (no real money)")
app.add_typer(paper_app, name="paper")


@paper_app.command("create")
def paper_create(name: str = "default", cash: float = 100_000.0, reset: bool = typer.Option(False, help="Reset if it exists")):
    from marketiq.data.db import init_db, session_scope
    from marketiq.paper.broker import PaperBroker

    init_db()
    with session_scope() as s:
        acct = PaperBroker.reset_account(s, name, cash) if reset else PaperBroker.create_account(s, name, cash)
        console.print(f"Paper account '{acct.name}' cash {acct.cash:,.2f}")


@paper_app.command("rebalance")
def paper_rebalance(name: str = "default", max_positions: int = 8, exit_score: float = 45.0,
                    as_of: Optional[str] = typer.Option(None)):
    """Apply the latest scan signals to the paper account."""
    from marketiq.data.db import init_db, session_scope
    from marketiq.ml.registry import load_predictor
    from marketiq.paper.broker import PaperBroker
    from marketiq.scanner import run_scan

    init_db()
    predictor, horizon, status = load_predictor()
    with session_scope() as s:
        d = date.fromisoformat(as_of) if as_of else None
        rows = run_scan(s, d, persist=True, ml_predictor=predictor, ml_horizon=horizon, ml_status=status)
        broker = PaperBroker(s, name)
        fills = broker.apply_signals(rows, max_positions=max_positions, exit_score=exit_score,
                                     trade_date=date.fromisoformat(rows[0].as_of) if rows else d)
        for f in fills:
            console.print(f"  {f.trade_date} {f.side:4} {f.symbol:6} {f.quantity:9.3f} @ {f.price:8.2f}  {f.reason}")
        pf = broker.portfolio(date.fromisoformat(rows[0].as_of) if rows else d)
    console.print(f"Equity {pf['equity']:,.2f}  cash {pf['cash']:,.2f}  return {pf['total_return'] * 100:+.2f}%  positions {len(pf['positions'])}")


@paper_app.command("status")
def paper_status(name: str = "default"):
    from marketiq.data.db import session_scope
    from marketiq.paper.broker import PaperBroker

    with session_scope() as s:
        pf = PaperBroker(s, name).portfolio()
    table = Table(title=f"Paper account {pf['account']}: equity {pf['equity']:,.2f} ({pf['total_return'] * 100:+.2f}%)")
    for col in ("Symbol", "Qty", "Avg cost", "Last", "Value", "P&L", "P&L %"):
        table.add_column(col, justify="right" if col != "Symbol" else "left")
    for p in pf["positions"]:
        table.add_row(p["symbol"], f"{p['quantity']:.3f}", f"{p['avg_cost']:.2f}", f"{p['last_price']:.2f}",
                      f"{p['market_value']:,.2f}", f"{p['unrealized_pnl']:+,.2f}", f"{p['unrealized_pnl_pct'] * 100:+.2f}%")
    console.print(table)


@app.command("coverage")
def coverage_cmd():
    """Show stored price coverage per symbol."""
    from marketiq.data import repository as repo
    from marketiq.data.db import session_scope

    with session_scope() as s:
        df = repo.price_coverage(s)
    table = Table(title="Price coverage")
    for c in df.columns:
        table.add_column(str(c))
    for row in df.itertuples(index=False):
        table.add_row(*[str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    app()
