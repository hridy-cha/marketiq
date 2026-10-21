from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marketiq.analysis.indicators import compute_indicators
from marketiq.config.loaders import ScoringConfig, load_scoring_config
from marketiq.scoring import scorer


def _cfg(**weights) -> ScoringConfig:
    base = load_scoring_config().model_dump()
    if weights:
        base["weights"] = weights
    return ScoringConfig(**base)


def test_linear_mapping_and_inversion():
    assert scorer.linear(0, -10, 10) == 50.0
    assert scorer.linear(10, -10, 10) == 100.0
    assert scorer.linear(-50, -10, 10) == 0.0
    assert scorer.linear(2, 5, 1) == 75.0  # inverted range: lower is better
    assert scorer.linear(None, 0, 1) is None


def test_weights_normalize_and_reject_unknown():
    cfg = _cfg(trend=2, momentum=2)
    assert cfg.weights == {"trend": 0.5, "momentum": 0.5}
    with pytest.raises(ValueError):
        _cfg(trend=1, bogus=1)
    with pytest.raises(ValueError):
        _cfg(trend=0, momentum=0)


def test_score_bounds_and_explanation_on_real_data(aapl_df, spy_df):
    cfg = load_scoring_config()
    ind, bench = compute_indicators(aapl_df), compute_indicators(spy_df)
    res = scorer.score_symbol("AAPL", ind, cfg, bench)
    assert 0 <= res.score <= 100
    assert res.as_of == "2024-12-31"
    assert set(res.components) == {"trend", "momentum", "volume", "volatility", "fundamentals", "sentiment", "ml"}
    available = [c for c in res.components.values() if c.available]
    assert {c.name for c in available} == {"trend", "momentum", "volume", "volatility"}
    assert sum(c.weight for c in available) == pytest.approx(1.0)
    assert all(c.weight == 0 for c in res.components.values() if not c.available)
    assert any("unavailable" in line.lower() for line in res.explanation)
    assert res.explanation[0].startswith("Composite score")
    # Every available factor carries a value/score/note.
    for c in available:
        assert c.factors
        for f in c.factors:
            assert f.note and (f.score is None or 0 <= f.score <= 100)


def test_composite_equals_weighted_mean_of_available_components(aapl_df):
    cfg = load_scoring_config()
    ind = compute_indicators(aapl_df)
    res = scorer.score_symbol("AAPL", ind, cfg)
    avail = {k: c for k, c in res.components.items() if c.available}
    total = sum(cfg.weights[k] for k in avail)
    expected = sum(cfg.weights[k] / total * c.score for k, c in avail.items())
    assert res.score == pytest.approx(round(expected, 1), abs=0.051)


def test_sentiment_and_ml_components_enter_when_available(aapl_df):
    cfg = load_scoring_config()
    ind = compute_indicators(aapl_df)
    agg = SimpleNamespace(available=True, score=0.5, reason="3 articles: 2 positive, 1 negative")
    res = scorer.score_symbol("AAPL", ind, cfg, sentiment_agg=agg, ml_prob=0.8, ml_horizon=20)
    assert res.components["sentiment"].score == pytest.approx(75.0)
    assert res.components["ml"].score == pytest.approx(80.0)
    assert res.components["sentiment"].available and res.components["ml"].available


def test_fundamentals_component_requires_three_factors():
    cfg = load_scoring_config()
    c = scorer.score_fundamentals({"pe_ratio": 80.0, "roe": None, "net_margin": 60.0}, {"pe_ratio": 20.0}, cfg)
    assert not c.available and "need 3" in c.note
    c2 = scorer.score_fundamentals({"pe_ratio": 80.0, "roe": 40.0, "net_margin": 60.0}, {}, cfg)
    assert c2.available and c2.score == pytest.approx(60.0)


def test_trend_component_full_bullish_stack():
    row = pd.Series({"close": 110, "sma_20": 105, "sma_50": 100, "sma_200": 90, "sma_50_slope_pct": 2.0, "high_52w": 112, "low_52w": 80})
    c = scorer.score_trend(row, load_scoring_config())
    by = {f.name: f for f in c.factors}
    assert by["sma_alignment"].score == 100.0
    assert by["position_in_52w_range"].score == pytest.approx((110 - 80) / (112 - 80) * 100)
    assert c.available and c.score > 80


def test_trend_component_needs_history():
    row = pd.Series({"close": 110, "sma_20": 105, "sma_50": np.nan, "sma_200": np.nan})
    c = scorer.score_trend(row, load_scoring_config())
    assert not c.available


def test_zero_weight_component_is_dropped(aapl_df):
    cfg = _cfg(trend=1, momentum=1, volume=1, volatility=1)
    res = scorer.score_symbol("AAPL", compute_indicators(aapl_df), cfg)
    assert set(res.components) == {"trend", "momentum", "volume", "volatility"}
