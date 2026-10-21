from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from marketiq.analysis.fundamentals import compute_fundamentals, universe_percentiles
from marketiq.analysis.sentiment import LexiconSentimentAnalyzer, aggregate_sentiment
from marketiq.data import repository as repo


def test_fundamentals_ttm_from_real_aapl(seeded_session, aapl_fundamentals):
    payloads = repo.get_fundamental_payloads(seeded_session, "AAPL", as_of=date(2024, 12, 31))
    snap = compute_fundamentals("AAPL", payloads, date(2024, 12, 31), price=250.0)
    income = sorted([r for r in aapl_fundamentals if r["statement"] == "income"], key=lambda r: r["date"])
    # Only statements filed by 2024-12-31 count: the Dec-2024 quarter was filed 2025-01-31, so the
    # latest usable quarter is Sep-2024.
    visible = [r for r in income if r["data"]["fillingDate"] <= "2024-12-31"]
    expected_rev = sum(r["data"]["revenue"] for r in visible[-4:])
    assert snap.ttm["revenue"] == pytest.approx(expected_rev)
    assert snap.last_period_end == date.fromisoformat(visible[-1]["date"])
    prev_rev = sum(r["data"]["revenue"] for r in visible[-8:-4])
    assert snap.factors["revenue_growth_yoy"] == pytest.approx((expected_rev - prev_rev) / prev_rev)
    assert snap.factors["pe_ratio"] == pytest.approx(250.0 / sum(r["data"]["epsdiluted"] for r in visible[-4:]))
    assert 0 < snap.factors["net_margin"] < 1
    assert snap.factors["debt_to_equity"] > 0
    assert snap.factors["current_ratio"] > 0


def test_fundamentals_stale_statements_ignored(seeded_session):
    payloads = repo.get_fundamental_payloads(seeded_session, "AAPL", as_of=date(2024, 12, 31))
    snap = compute_fundamentals("AAPL", payloads, date(2027, 1, 1), price=250.0, max_staleness_days=400)
    assert snap.factors == {}
    assert any("older than" in n for n in snap.notes)


def test_fundamentals_empty():
    snap = compute_fundamentals("X", [], date(2024, 1, 1), price=10.0)
    assert snap.factors == {} and "No financial statements" in snap.notes[0]


def test_universe_percentiles_direction():
    S = SimpleNamespace
    snaps = {
        "A": S(factors={"pe_ratio": 10.0, "roe": 0.3}),
        "B": S(factors={"pe_ratio": 20.0, "roe": 0.2}),
        "C": S(factors={"pe_ratio": 30.0, "roe": 0.1}),
        "D": S(factors={"pe_ratio": None, "roe": 0.05}),
    }
    pct = universe_percentiles(snaps, {"pe_ratio": -1, "roe": 1})
    assert pct["A"]["pe_ratio"] == 100.0 and pct["C"]["pe_ratio"] < pct["B"]["pe_ratio"]  # lower P/E ranks higher
    assert pct["A"]["roe"] == 100.0 and pct["D"]["roe"] == 25.0
    assert pct["D"]["pe_ratio"] is None


def test_lexicon_sentiment_labels_and_negation():
    a = LexiconSentimentAnalyzer()
    assert a.analyze("Company beats estimates, raises guidance on strong demand").label == "positive"
    assert a.analyze("Shares plunge after earnings miss and downgrade").label == "negative"
    assert a.analyze("The company will hold its annual meeting on Tuesday").label == "neutral"
    assert a.analyze("Revenue did not decline this quarter").score > 0
    assert a.analyze("").score == 0.0


def test_aggregate_sentiment_recency_weighting_and_min_articles():
    now = datetime(2025, 1, 10, 12)
    S = SimpleNamespace
    arts = [
        S(published_at=now - timedelta(days=0), sentiment_score=1.0, sentiment_label="positive"),
        S(published_at=now - timedelta(days=10), sentiment_score=-1.0, sentiment_label="negative"),
    ]
    agg = aggregate_sentiment(arts, now, half_life_days=5, min_articles=2)
    assert agg.available and agg.article_count == 2
    # weight of the old article is 0.25, so (1*1 + 0.25*-1) / 1.25 = 0.6
    assert agg.score == pytest.approx(0.6)
    assert agg.positive == 1 and agg.negative == 1
    few = aggregate_sentiment(arts[:1], now, 5, min_articles=2)
    assert not few.available and few.score is None
