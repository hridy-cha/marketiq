import json

import numpy as np
import pandas as pd
import pytest

from marketiq.analysis.indicators import compute_indicators
from marketiq.ml.features import FEATURE_COLUMNS, build_dataset, make_features, make_labels
from marketiq.ml.model import DirectionModel, walk_forward


@pytest.fixture(scope="module")
def panel(aapl_df, msft_df, spy_df):
    ind = {"AAPL": compute_indicators(aapl_df), "MSFT": compute_indicators(msft_df)}
    return build_dataset(ind, horizon=20, threshold=0.0, bench_ind=compute_indicators(spy_df))


def test_labels_use_only_future_bars(aapl_df):
    ind = compute_indicators(aapl_df)
    label, fwd = make_labels(ind, horizon=20)
    assert label.iloc[-20:].isna().all()
    t = 100
    expected = ind["adj_close"].iloc[t + 20] / ind["adj_close"].iloc[t] - 1.0
    assert fwd.iloc[t] == pytest.approx(expected)
    assert label.iloc[t] == float(expected > 0)


def test_features_have_no_lookahead(aapl_df, spy_df):
    bench = compute_indicators(spy_df)
    full = make_features(compute_indicators(aapl_df), bench)
    cut = make_features(compute_indicators(aapl_df.iloc[:400]), bench.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], cut, check_exact=False, rtol=1e-9)


def test_dataset_shape_and_columns(panel):
    assert set(panel.columns) == set(FEATURE_COLUMNS) | {"label", "fwd_return", "symbol"}
    assert panel[FEATURE_COLUMNS].notna().all().all()
    assert set(panel["symbol"]) == {"AAPL", "MSFT"}
    assert panel.index.is_monotonic_increasing
    assert panel["label"].isin([0.0, 1.0]).all()


def test_walk_forward_gap_prevents_label_leakage(panel):
    report = walk_forward(panel, horizon_days=20, n_folds=2, min_train_rows=200, test_months=3)
    assert report.folds, "expected at least one fold on the 2022-2024 fixture"
    dates = pd.DatetimeIndex(sorted(panel.index.unique()))
    for f in report.folds:
        train_end, test_start = pd.Timestamp(f.train_end), pd.Timestamp(f.test_start)
        # At least `horizon` trading days between the last training bar and the first test bar,
        # so the last training label (which looks 20 bars ahead) ends before the test window.
        gap = dates.searchsorted(test_start) - dates.searchsorted(train_end)
        assert gap >= 20
        assert 0 <= f.accuracy <= 1 and 0 <= f.base_rate <= 1
    assert report.pooled["n_folds"] == len(report.folds)
    assert "auc" in report.pooled


def test_model_fit_predict_roundtrip(panel):
    m = DirectionModel(20).fit(panel)
    p = m.predict_proba(panel.tail(10))
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()
    imp = m.feature_importance()
    assert set(imp) == set(FEATURE_COLUMNS) and sum(imp.values()) == pytest.approx(1.0)


def test_registry_train_and_gate(seeded_session, tmp_path, monkeypatch):
    """Train on the fixture database, then verify the registry and the validation gate."""
    from marketiq.config.loaders import Universe, UniverseTicker
    from marketiq.config.settings import get_settings
    from marketiq.ml import registry

    monkeypatch.setenv("DATA_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ML_MIN_TRAIN_ROWS", "200")
    get_settings.cache_clear()
    monkeypatch.setattr(registry, "load_universe", lambda: Universe(
        benchmark="SPY", tickers=[UniverseTicker(symbol="AAPL"), UniverseTicker(symbol="MSFT")]))
    meta = registry.train_and_register(seeded_session, name="test_model", horizon_days=20, n_folds=2)
    assert meta["n_train_rows"] > 200 and meta["backend"] in {"lightgbm", "sklearn_hgb"}
    assert (tmp_path / "models" / "test_model" / meta["version"] / "model.pkl").exists()
    saved = json.load(open(tmp_path / "models" / "test_model" / meta["version"] / "metadata.json"))
    assert saved["walk_forward"]["pooled"]["n_folds"] >= 1
    model, loaded_meta = registry.load_latest("test_model")
    assert loaded_meta["version"] == meta["version"] and model.horizon_days == 20

    # Gate: with a strict threshold the predictor is refused and the reason names the metrics.
    monkeypatch.setattr(registry, "load_scoring_config", lambda: type("C", (), {"ml": type("M", (), {
        "model_name": "test_model", "min_walk_forward_auc": 0.99, "min_walk_forward_balanced_accuracy": 0.99})()})())
    pred, horizon, status = registry.load_predictor("test_model")
    assert pred is None and horizon is None and "failed the validation gate" in status
    # With the gate off, predictions come back as probabilities.
    pred, horizon, status = registry.load_predictor("test_model", enforce_gate=False)
    from marketiq.data import repository as repo

    ind = compute_indicators(repo.get_price_frame(seeded_session, "AAPL"))
    bench = compute_indicators(repo.get_price_frame(seeded_session, "SPY"))
    p = pred("AAPL", ind, bench)
    assert horizon == 20 and 0.0 <= p <= 1.0
    get_settings.cache_clear()
