"""
Import the FinRL-X bundled SQLite database (data/finrl_trading.db) into MarketIQ's schema.

Source tables (FinRL-X `DataStore`): price_data(ticker, date, open, high, low, close, adj_close,
volume) and raw_payloads(source, payload, ticker, date, row_json) where payload is one of
income|balance|cashflow|ratios|profile. Only the requested symbols are copied.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from marketiq.data import repository as repo

logger = logging.getLogger(__name__)


def _parse_date(v) -> Optional[date]:
    try:
        ts = pd.to_datetime(v, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception:
        return None


def import_prices(session: Session, finrlx_db: Path, symbols: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with sqlite3.connect(finrlx_db) as conn:
        for sym in symbols:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, adj_close, volume FROM price_data WHERE ticker = ? ORDER BY date",
                conn,
                params=(sym,),
            )
            if df.empty:
                logger.warning("No FinRL-X price rows for %s", sym)
                counts[sym] = 0
                continue
            counts[sym] = repo.upsert_price_bars(session, df, sym, source="finrlx")
    return counts


def import_fundamentals(session: Session, finrlx_db: Path, symbols: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with sqlite3.connect(finrlx_db) as conn:
        for sym in symbols:
            rows = pd.read_sql_query(
                "SELECT payload, date, row_json FROM raw_payloads WHERE ticker = ? AND payload IN "
                "('income','balance','cashflow','ratios','profile')",
                conn,
                params=(sym,),
            )
            records: List[dict] = []
            for r in rows.itertuples(index=False):
                try:
                    data = json.loads(r.row_json)
                except json.JSONDecodeError:
                    continue
                period_end = _parse_date(data.get("date") or r.date)
                if period_end is None:
                    continue
                filing = _parse_date(data.get("fillingDate") or data.get("filingDate") or data.get("acceptedDate"))
                records.append(
                    {
                        "symbol": sym,
                        "statement": r.payload,
                        "period_end": period_end,
                        "filing_date": filing,
                        "source": "finrlx_fmp",
                        "data": data,
                    }
                )
            counts[sym] = repo.upsert_fundamental_payloads(session, records)
    return counts


def import_all(session: Session, finrlx_db: Path, symbols: Iterable[str]) -> Dict[str, Dict[str, int]]:
    symbols = list(symbols)
    if not Path(finrlx_db).exists():
        raise FileNotFoundError(
            f"{finrlx_db} not found. Extract data/finrl_trading.7z first (see README: Data setup)."
        )
    prices = import_prices(session, finrlx_db, symbols)
    fundamentals = import_fundamentals(session, finrlx_db, symbols)
    return {"prices": prices, "fundamentals": fundamentals}
