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
    predictor, horizon = (None, None) if no_ml else load_predictor()
    with session_scope() as s:
        rows = run_scan(s, date.fromisoformat(as_of) if as_of else None, persist=not no_persist,
                        ml_predictor=predictor, ml_horizon=horizon)
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
