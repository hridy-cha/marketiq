from datetime import date

from marketiq.config.loaders import Universe, UniverseTicker, load_scoring_config, load_signal_config
from marketiq.data import repository as repo
from marketiq.scanner import ScanContext, run_scan

UNIVERSE = Universe(benchmark="SPY", tickers=[UniverseTicker(symbol="AAPL", name="Apple"), UniverseTicker(symbol="MSFT", name="Microsoft")])


def test_scan_context_evaluates_as_of_and_latest(seeded_session):
    ctx = ScanContext.load(seeded_session, universe=UNIVERSE, with_news=False)
    latest = ctx.evaluate()
    assert [r.symbol for r in latest] and all(r.as_of == "2024-12-31" for r in latest)
    assert latest == sorted(latest, key=lambda r: r.score.score, reverse=True)
    past = ctx.evaluate(as_of=date(2023, 6, 30))
    assert all(r.as_of == "2023-06-30" for r in past)
    # Point-in-time: the June-2023 evaluation must not see later price bars.
    assert all(r.indicators["close"] == float(ctx.prices[r.symbol].loc["2023-06-30", "close"]) for r in past)
    aapl = next(r for r in past if r.symbol == "AAPL")
    assert aapl.fundamentals.last_filing_date <= date(2023, 6, 30)
    assert aapl.signal.signal in {"BUY", "SELL", "HOLD", "WATCH"}
    assert aapl.score.explanation


def test_run_scan_persists_snapshots(seeded_session, monkeypatch):
    monkeypatch.setattr("marketiq.scanner.load_universe", lambda: UNIVERSE)
    rows = run_scan(seeded_session, persist=True, with_news=False)
    assert len(rows) == 2
    scores = repo.latest_scores(seeded_session)
    assert {s.symbol for s in scores} == {"AAPL", "MSFT"}
    assert scores[0].as_of == date(2024, 12, 31)
    sigs = repo.latest_signals(seeded_session)
    assert set(sigs) == {"AAPL", "MSFT"} and all(s.reasons for s in sigs.values())
    # Re-running the same day replaces rather than duplicates.
    run_scan(seeded_session, persist=True, with_news=False)
    assert len(repo.latest_scores(seeded_session)) == 2
