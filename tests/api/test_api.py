"""API tests against a seeded SQLite fixture with the FastAPI TestClient. No network."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from marketiq.config.loaders import Universe, UniverseTicker
from marketiq.scanner import run_scan

UNIVERSE = Universe(benchmark="SPY", tickers=[UniverseTicker(symbol="AAPL", name="Apple", sector="Technology"),
                                              UniverseTicker(symbol="MSFT", name="Microsoft", sector="Technology")])


@pytest.fixture()
def client(seeded_session, monkeypatch, tmp_path):
    import marketiq.api.routers.ops as ops
    import marketiq.api.routers.stocks as stocks
    import marketiq.scanner as scanner

    monkeypatch.setattr(stocks, "cached_universe", lambda: UNIVERSE)
    monkeypatch.setattr(ops, "cached_universe", lambda: UNIVERSE)
    monkeypatch.setattr(scanner, "load_universe", lambda: UNIVERSE)
    monkeypatch.setenv("DATA_MODELS_DIR", str(tmp_path / "models"))
    from marketiq.config.settings import get_settings

    get_settings.cache_clear()
    run_scan(seeded_session, persist=True, with_news=False)
    seeded_session.commit()
    from marketiq.api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_health_and_universe(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["symbols_with_prices"] == 3 and h["latest_scan"] == "2024-12-31"
    assert h["live_trading_enabled"] is False
    u = client.get("/api/universe").json()
    assert [t["symbol"] for t in u["tickers"]] == ["AAPL", "MSFT"] and u["benchmark"] == "SPY"


def test_rankings_and_score_detail(client):
    r = client.get("/api/rankings").json()
    assert r["as_of"] == "2024-12-31" and len(r["rows"]) == 2
    top = r["rows"][0]
    assert top["rank"] == 1 and top["signal"] in {"BUY", "SELL", "HOLD", "WATCH"} and 0 <= top["score"] <= 100
    assert set(top["components"]) >= {"trend", "momentum"}
    s = client.get(f"/api/stocks/{top['symbol']}/score").json()
    assert s["score"] == top["score"] and s["explanation"][0].startswith("Composite score")
    assert s["signal"]["reasons"] and s["history"][-1]["score"] == top["score"]
    assert client.get("/api/stocks/ZZZZ/score").status_code == 404


def test_bars_indicators_fundamentals(client):
    b = client.get("/api/stocks/aapl/bars?limit=50").json()
    assert b["symbol"] == "AAPL" and len(b["bars"]) == 50 and b["bars"][-1]["date"] == "2024-12-31"
    i = client.get("/api/stocks/AAPL/indicators?limit=100").json()
    assert len(i["dates"]) == 100 and set(i["series"]) >= {"sma_20", "rsi_14", "macd"} and "trend_regime" in i["patterns"]
    assert i["latest"]["rsi_14"] is not None
    f = client.get("/api/stocks/AAPL/fundamentals").json()
    assert f["factors"]["pe_ratio"] > 0 and len(f["quarters"]) == 8
    assert f["last_filing_date"] <= "2024-12-31"
    n = client.get("/api/stocks/AAPL/news").json()
    assert n["articles"] == [] and n["aggregate"]["available"] is False


def test_signals_filter(client):
    all_rows = client.get("/api/signals").json()["rows"]
    assert len(all_rows) == 2
    sig = all_rows[0]["signal"]
    filtered = client.get(f"/api/signals?signal={sig}").json()["rows"]
    assert all(r["signal"] == sig for r in filtered) and filtered


def test_watchlist_crud(client):
    assert client.post("/api/watchlist", json={"symbol": "aapl", "note": "hi"}).status_code == 201
    assert client.post("/api/watchlist", json={"symbol": "ZZZZ"}).status_code == 404
    w = client.get("/api/watchlist").json()["items"]
    assert w[0]["symbol"] == "AAPL" and w[0]["score"] is not None and w[0]["signal"]
    assert client.delete("/api/watchlist/AAPL").json()["removed"] == "AAPL"
    assert client.delete("/api/watchlist/AAPL").status_code == 404


def test_config_and_weight_update(client, tmp_path, monkeypatch):
    import shutil

    from marketiq.config.settings import get_settings

    src = get_settings().data.path("scoring_file")
    dst = tmp_path / "scoring.yaml"
    shutil.copy(src, dst)
    monkeypatch.setenv("DATA_SCORING_FILE", str(dst))
    get_settings.cache_clear()
    c = client.get("/api/config").json()
    assert abs(sum(c["scoring"]["weights"].values()) - 1) < 1e-9
    r = client.put("/api/config/weights", json={"weights": {"trend": 1, "momentum": 1}})
    assert r.status_code == 200 and r.json()["weights"] == {"trend": 0.5, "momentum": 0.5}
    assert client.put("/api/config/weights", json={"weights": {"bogus": 1}}).status_code == 400
    get_settings.cache_clear()


def test_paper_trading_flow(client):
    assert client.post("/api/paper/accounts", json={"name": "api", "starting_cash": 50000}).status_code == 201
    o = client.post("/api/paper/accounts/api/orders", json={"symbol": "AAPL", "side": "BUY", "notional": 10000})
    assert o.status_code == 201 and o.json()["side"] == "BUY"
    pf = client.get("/api/paper/accounts/api").json()
    assert pf["positions"][0]["symbol"] == "AAPL" and abs(pf["equity"] - 50000) < 50
    bad = client.post("/api/paper/accounts/api/orders", json={"symbol": "MSFT", "side": "SELL"})
    assert bad.status_code == 400
    rb = client.post("/api/paper/accounts/api/rebalance?max_positions=2").json()
    assert rb["as_of"] == "2024-12-31" and "portfolio" in rb
    assert client.get("/api/paper/accounts/nope").status_code == 404


def test_ml_endpoints_without_model(client):
    m = client.get("/api/ml/models").json()
    assert m["latest"] is None and m["gate"]["passed"] is False
    p = client.get("/api/ml/predictions").json()
    assert p["available"] is False


def test_backtest_endpoint_small_window(client):
    r = client.post("/api/backtests", json={"start": "2024-06-03", "end": "2024-09-30", "max_positions": 2,
                                           "use_ml": False, "name": "api-test"})
    assert r.status_code == 201
    run_id = r.json()["id"]
    runs = client.get("/api/backtests").json()["runs"]
    assert runs[0]["id"] == run_id and runs[0]["name"] == "api-test"
    detail = client.get(f"/api/backtests/{run_id}").json()
    assert detail["equity_curve"][0]["equity"] == 100000.0 and "total_return" in detail["metrics"]
    assert detail["config"]["ml_status"] == "ML disabled for this run"
    assert client.get("/api/backtests/9999").status_code == 404
