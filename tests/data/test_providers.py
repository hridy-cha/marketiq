"""Provider tests with recorded/stubbed payloads. No network access is used."""

from datetime import date

import pandas as pd
import pytest

from marketiq.data.providers.base import normalize_price_frame
from marketiq.data.providers.fmp import FMPProvider


def test_normalize_price_frame_handles_yahoo_columns():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Adj Close": [1.1, 2.1],
            "Volume": [100, 200],
        }
    )
    out = normalize_price_frame(df)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert out["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["adj_close"].tolist() == [2.1, 1.1]


def test_normalize_price_frame_without_adj_close_uses_close():
    df = pd.DataFrame({"date": ["2024-01-02"], "open": [1], "high": [1], "low": [1], "close": [5.0], "volume": [1]})
    assert normalize_price_frame(df)["adj_close"].iloc[0] == 5.0


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Http:
    """Fake requests.Session returning canned FMP responses per endpoint."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for key, payload in self.routes.items():
            if url.endswith(key):
                return _Resp(payload)
        return _Resp([])


def test_fmp_prices_and_fundamentals_parse():
    routes = {
        "historical-price-eod/full": {
            "symbol": "AAPL",
            "historical": [
                {"date": "2024-01-03", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "adjClose": 1.4, "volume": 10},
                {"date": "2024-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.6, "adjClose": 1.5, "volume": 11},
            ],
        },
        "income-statement": [{"date": "2023-12-30", "revenue": 100.0, "netIncome": 10.0, "fillingDate": "2024-02-01"}],
        "balance-sheet-statement": [{"date": "2023-12-30", "totalStockholdersEquity": 50.0}],
        "cash-flow-statement": [{"date": "2023-12-30", "freeCashFlow": 20.0}],
        "ratios": [{"date": "2023-12-30", "priceEarningsRatio": 25.0}],
        "earnings": [{"date": "2024-05-02", "epsEstimated": 1.5, "epsActual": None}],
    }
    http = _Http(routes)
    p = FMPProvider(api_key="test-key", session=http)
    prices = p.fetch_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert len(prices) == 2 and prices["date"].is_monotonic_increasing
    assert prices["adj_close"].tolist() == [1.5, 1.4]
    bundle = p.fetch_fundamentals("AAPL")
    assert set(bundle.statements) == {"income", "balance", "cashflow", "ratios"}
    assert bundle.statements["income"][0]["revenue"] == 100.0
    assert bundle.earnings[0]["event_date"] == date(2024, 5, 2)
    # API key is sent as a query parameter, never embedded in the URL path.
    assert all(params.get("apikey") == "test-key" for _, params in http.calls)


def test_fmp_requires_key(monkeypatch):
    from marketiq.config.settings import get_settings

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(Exception):
        FMPProvider()
    get_settings.cache_clear()
