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
