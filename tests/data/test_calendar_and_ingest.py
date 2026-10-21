from datetime import date

import pandas as pd

from marketiq.data import ingest
from marketiq.data import repository as repo
from marketiq.data.calendar import consolidate_ranges, is_trading_day, missing_trading_days, trading_days


def test_trading_days_known_month():
    days = trading_days("2024-01-01", "2024-01-31")
    assert len(days) == 21
    assert not is_trading_day("2024-01-01")  # New Year's Day
    assert not is_trading_day("2024-01-15")  # MLK Day
    assert is_trading_day("2024-01-02")


def test_missing_days_ignores_future_and_weekends():
    have = ["2024-01-02", "2024-01-03", "2024-01-08"]
    missing = missing_trading_days(have, "2024-01-01", "2024-01-10")
    assert missing == ["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"]


def test_consolidate_ranges_merges_weekend_gaps():
    days = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-02-01"]
    assert consolidate_ranges(days) == [("2024-01-04", "2024-01-09"), ("2024-02-01", "2024-02-01")]
    assert consolidate_ranges([]) == []


class StubPriceProvider:
    """Serves bars from a fixture frame and records the ranges it was asked for."""

    name = "stub"

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index()
        self.calls = []

    def fetch_prices(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        m = (self.df["date"].dt.date >= start) & (self.df["date"].dt.date <= end)
        return self.df.loc[m]


def test_fetch_missing_prices_only_requests_gaps(db_session, aapl_df):
    # Store Jan-Jun 2023, then ask for all of 2023: provider should only be asked for Jul-Dec.
    half = aapl_df.loc["2023-01-01":"2023-06-30"]
    repo.upsert_price_bars(db_session, half.reset_index(), "AAPL")
    provider = StubPriceProvider(aapl_df)
    res = ingest.fetch_missing_prices(db_session, provider, ["AAPL"], date(2023, 1, 1), end=date(2023, 12, 31))
    assert len(provider.calls) == 1
    sym, start, end = provider.calls[0]
    assert start == date(2023, 7, 3) and end == date(2023, 12, 29)
    stored = repo.get_price_frame(db_session, "AAPL", start=date(2023, 1, 1), end=date(2023, 12, 31))
    assert len(stored) == 250  # NYSE trading days in 2023
    assert res["AAPL"] == 250 - len(half)
    # Second run: nothing missing, provider not called.
    res2 = ingest.fetch_missing_prices(db_session, provider, ["AAPL"], date(2023, 1, 1), end=date(2023, 12, 31))
    assert res2["AAPL"] == 0 and len(provider.calls) == 1
