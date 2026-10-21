"""Shared fixtures. Tests run against an isolated SQLite database seeded from real price slices."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def aapl_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "AAPL_daily.csv", parse_dates=["date"])
    return df.set_index("date")


@pytest.fixture(scope="session")
def msft_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "MSFT_daily.csv", parse_dates=["date"])
    return df.set_index("date")


@pytest.fixture(scope="session")
def spy_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "SPY_daily.csv", parse_dates=["date"])
    return df.set_index("date")


@pytest.fixture(scope="session")
def aapl_fundamentals() -> list:
    with open(FIXTURES / "AAPL_fundamentals.json") as fh:
        return json.load(fh)


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    """Fresh SQLite database per test, with the ORM engine pointed at it."""
    from marketiq.data import db as dbmod

    url = f"sqlite:///{tmp_path / 'test.db'}"
    dbmod.configure_engine(url)
    dbmod.init_db()
    with dbmod.session_scope() as s:
        yield s


@pytest.fixture()
def seeded_session(db_session, aapl_df, msft_df, spy_df, aapl_fundamentals):
    """Database with AAPL, MSFT and SPY bars (2022-2024) and AAPL fundamentals."""
    from marketiq.data import repository as repo

    repo.upsert_tickers(
        db_session,
        [
            {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
            {"symbol": "MSFT", "name": "Microsoft", "sector": "Technology"},
            {"symbol": "SPY", "name": "Benchmark", "is_benchmark": True},
        ],
    )
    for sym, df in (("AAPL", aapl_df), ("MSFT", msft_df), ("SPY", spy_df)):
        repo.upsert_price_bars(db_session, df.reset_index(), sym, source="fixture")
    rows = []
    for r in aapl_fundamentals:
        rows.append(
            {
                "symbol": "AAPL",
                "statement": r["statement"],
                "period_end": date.fromisoformat(r["date"]),
                "filing_date": pd.to_datetime(r["data"].get("fillingDate")).date()
                if r["data"].get("fillingDate")
                else None,
                "source": "fixture",
                "data": r["data"],
            }
        )
    repo.upsert_fundamental_payloads(db_session, rows)
    db_session.commit()
    return db_session
