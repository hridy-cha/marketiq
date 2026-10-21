"""
Model registry: trains, saves and loads direction models.

Artifacts live under `models/<name>/<version>/` as `model.pkl` plus `metadata.json` (features,
horizon, threshold, training range, walk-forward metrics, backend). `latest.json` points at the
most recent version. `load_predictor` returns a callable the scanner can use, or (None, None)
when no model has been trained.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from marketiq.config.loaders import load_scoring_config, load_universe
from marketiq.config.settings import get_settings
from marketiq.ml.features import build_dataset, make_features
from marketiq.ml.model import DirectionModel, walk_forward

logger = logging.getLogger(__name__)


def _models_dir() -> Path:
    return get_settings().data.path("models_dir")


def train_and_register(
    session: Session,
    name: Optional[str] = None,
    horizon_days: Optional[int] = None,
    threshold: Optional[float] = None,
    start: Optional[str] = None,
    n_folds: int = 5,
) -> Dict:
    """Build the dataset from stored prices, run walk-forward evaluation, fit the final model on all
    rows, and save it. Returns the metadata dict (which includes the walk-forward report)."""
    from marketiq.analysis.indicators import compute_indicators
    from marketiq.data import repository as repo

    s = get_settings()
    name = name or load_scoring_config().ml.model_name
    horizon_days = horizon_days or s.ml.horizon_days
    threshold = s.ml.return_threshold if threshold is None else threshold
    universe = load_universe()

    indicators = {}
    for sym in universe.all_symbols:
        df = repo.get_price_frame(session, sym, start=pd.Timestamp(start).date() if start else None)
        if not df.empty:
            indicators[sym] = compute_indicators(df)
    bench = indicators.get(universe.benchmark)
    panel = build_dataset({k: v for k, v in indicators.items() if k != universe.benchmark}, horizon_days, threshold, bench)
    if len(panel) < s.ml.min_train_rows:
        raise ValueError(f"Only {len(panel)} training rows; need at least {s.ml.min_train_rows}")

    report = walk_forward(panel, horizon_days, threshold, n_folds=n_folds, min_train_rows=s.ml.min_train_rows)
    model = DirectionModel(horizon_days, threshold).fit(panel)
    report.feature_importance = model.feature_importance()

    version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = _models_dir() / name / version
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "name": name,
        "version": version,
        "backend": model.backend,
        "horizon_days": horizon_days,
        "threshold": threshold,
        "features": model.features,
        "train_start": model.train_start,
        "train_end": model.train_end,
        "n_train_rows": model.n_train,
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "walk_forward": report.to_dict(),
        "path": str(out_dir / "model.pkl"),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(_models_dir() / name / "latest.json", "w") as fh:
        json.dump({"version": version, "path": str(out_dir)}, fh)

    from marketiq.data.models import MLModelRecord

    session.add(MLModelRecord(
        name=name, version=version, horizon_days=horizon_days, threshold=threshold,
        train_start=pd.Timestamp(model.train_start).date(), train_end=pd.Timestamp(model.train_end).date(),
        metrics=report.pooled, features=model.features, path=str(out_dir / "model.pkl"),
    ))
    session.commit()
    logger.info("Registered model %s/%s (%s) trained on %d rows", name, version, model.backend, model.n_train)
    return meta


def load_latest(name: Optional[str] = None) -> Tuple[Optional[DirectionModel], Optional[dict]]:
    name = name or load_scoring_config().ml.model_name
    pointer = _models_dir() / name / "latest.json"
    if not pointer.exists():
        return None, None
    with open(pointer) as fh:
        ptr = json.load(fh)
    version_dir = Path(ptr["path"])
    try:
        with open(version_dir / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        with open(version_dir / "metadata.json") as fh:
            meta = json.load(fh)
        return model, meta
    except (OSError, pickle.UnpicklingError) as exc:
        logger.warning("Could not load model %s: %s", version_dir, exc)
        return None, None


def validation_gate(meta: dict, cfg=None) -> Tuple[bool, str]:
    """Check the model's pooled walk-forward metrics against config/scoring.yaml `ml` thresholds."""
    cfg = cfg or load_scoring_config().ml
    pooled = (meta.get("walk_forward") or {}).get("pooled") or {}
    auc, bal = pooled.get("auc"), pooled.get("balanced_accuracy")
    label = f"{meta.get('name')}/{meta.get('version')}"
    if auc is None or bal is None:
        return False, f"{label} has no walk-forward metrics"
    if auc < cfg.min_walk_forward_auc or bal < cfg.min_walk_forward_balanced_accuracy:
        return False, (f"{label} failed the validation gate: walk-forward AUC {auc:.3f} "
                       f"(min {cfg.min_walk_forward_auc}), balanced accuracy {bal:.3f} "
                       f"(min {cfg.min_walk_forward_balanced_accuracy})")
    return True, f"{label} passed validation (AUC {auc:.3f}, balanced accuracy {bal:.3f})"


def load_predictor(name: Optional[str] = None, enforce_gate: bool = True) -> Tuple[Optional[Callable], Optional[int], str]:
    """Return (predictor(symbol, ind, bench_ind) -> prob_up, horizon, status message).
    predictor is None when no model exists or the latest one fails the validation gate."""
    model, meta = load_latest(name)
    if model is None:
        return None, None, "No trained model (run `marketiq train`)"
    ok, msg = validation_gate(meta)
    if enforce_gate and not ok:
        logger.warning("ML component disabled: %s", msg)
        return None, None, msg

    def predict(symbol: str, ind: pd.DataFrame, bench_ind: Optional[pd.DataFrame] = None) -> Optional[float]:
        feats = make_features(ind, bench_ind).iloc[[-1]]
        if feats[model.features].isna().any(axis=1).iloc[0]:
            return None
        return float(model.predict_proba(feats)[0])

    return predict, int(meta["horizon_days"]), msg
