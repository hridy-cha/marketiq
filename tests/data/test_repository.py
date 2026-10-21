from datetime import date

import pandas as pd

from marketiq.data import repository as repo


def test_upsert_and_read_prices(db_session, aapl_df):
    n = repo.upsert_price_bars(db_session, aapl_df.reset_index(), "AAPL", source="fixture")
    assert n == len(aapl_df) == 753
    out = repo.get_price_frame(db_session, "AAPL")
    assert len(out) == 753
    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert out.index.is_monotonic_increasing
    # Real values from the bundled dataset, first row of the 2022 slice.
    first = out.iloc[0]
    assert out.index[0] == pd.Timestamp("2022-01-03")
    assert abs(first["close"] - aapl_df.iloc[0]["close"]) < 1e-9


def test_upsert_is_idempotent_and_updates(db_session, aapl_df):
    repo.upsert_price_bars(db_session, aapl_df.reset_index(), "AAPL")
    changed = aapl_df.reset_index().head(1).copy()
    changed["close"] = 999.0
    repo.upsert_price_bars(db_session, changed, "AAPL", source="test")
    out = repo.get_price_frame(db_session, "AAPL")
    assert len(out) == 753
    assert out.iloc[0]["close"] == 999.0


def test_date_range_filter(seeded_session):
    out = repo.get_price_frame(seeded_session, "AAPL", start=date(2023, 1, 1), end=date(2023, 1, 31))
    assert len(out) == 20  # NYSE trading days in Jan 2023
    assert out.index.min() == pd.Timestamp("2023-01-03")


def test_empty_symbol_returns_empty_frame(db_session):
    out = repo.get_price_frame(db_session, "ZZZZ")
    assert out.empty
    assert "close" in out.columns


def test_fundamental_payloads_point_in_time(seeded_session):
    rows = repo.get_fundamental_payloads(seeded_session, "AAPL", statement="income")
    assert len(rows) == 12
    assert rows[0].period_end < rows[-1].period_end
    # As of mid-2023, only statements filed on or before that date are visible.
    pit = repo.get_fundamental_payloads(seeded_session, "AAPL", statement="income", as_of=date(2023, 6, 30))
    assert 0 < len(pit) < 12
    assert all((r.filing_date or r.period_end) <= date(2023, 6, 30) for r in pit)


def test_watchlist_roundtrip(db_session):
    repo.add_watchlist(db_session, "AAPL", note="test")
    repo.add_watchlist(db_session, "AAPL", note="updated")
    items = repo.list_watchlist(db_session)
    assert [i.symbol for i in items] == ["AAPL"]
    assert items[0].note == "updated"
    assert repo.remove_watchlist(db_session, "AAPL")
    assert repo.list_watchlist(db_session) == []


def test_score_and_signal_snapshots(db_session):
    repo.save_score(db_session, symbol="AAPL", as_of=date(2024, 1, 2), score=61.2, components={}, explanation=["x"])
    repo.save_score(db_session, symbol="AAPL", as_of=date(2024, 1, 3), score=66.0, components={}, explanation=["y"])
    repo.save_score(db_session, symbol="AAPL", as_of=date(2024, 1, 3), score=70.0, components={}, explanation=["z"])
    latest = repo.latest_scores(db_session)
    assert len(latest) == 1 and latest[0].score == 70.0
    prev = repo.previous_score(db_session, "AAPL", date(2024, 1, 3))
    assert prev.score == 61.2
    repo.save_signal(db_session, symbol="AAPL", as_of=date(2024, 1, 3), signal="BUY", score=70.0, reasons=["r"])
    assert repo.latest_signals(db_session)["AAPL"].signal == "BUY"
