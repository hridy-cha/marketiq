"""
Direction classifier and walk-forward evaluation.

Model: LightGBM gradient boosting when installed, otherwise scikit-learn HistGradientBoosting. Both
are tabular, handle missing values, and train in seconds on this universe.

Evaluation: expanding-window walk-forward. For fold k the model trains on rows with
date <= train_end_k and is scored on rows with date in (train_end_k + horizon, test_end_k]. The gap
of `horizon` bars after train_end guarantees no training label overlaps the test window. Metrics
reported per fold and pooled: accuracy, balanced accuracy, ROC AUC, precision in the top decile of
predicted probability, and the mean forward return of that top decile versus all rows.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from marketiq.ml.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _make_estimator(seed: int = 42):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1,
        ), "lightgbm"
    except (ImportError, OSError) as exc:  # not installed, or macOS wheel missing libomp
        logger.info("LightGBM unavailable (%s); using scikit-learn HistGradientBoosting", type(exc).__name__)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                              min_samples_leaf=50, l2_regularization=1.0, random_state=seed), "sklearn_hgb"


@dataclass
class FoldMetrics:
    fold: int
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    auc: Optional[float]
    top_decile_precision: Optional[float]
    top_decile_mean_fwd_return: Optional[float]
    all_mean_fwd_return: float
    base_rate: float


@dataclass
class WalkForwardReport:
    horizon_days: int
    threshold: float
    folds: List[FoldMetrics]
    pooled: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"horizon_days": self.horizon_days, "threshold": self.threshold,
                "folds": [asdict(f) for f in self.folds], "pooled": self.pooled,
                "feature_importance": self.feature_importance}


def _metrics(y: np.ndarray, p: np.ndarray, fwd: np.ndarray) -> Dict[str, Any]:
    pred = (p >= 0.5).astype(int)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "base_rate": float(y.mean()),
        "all_mean_fwd_return": float(np.nanmean(fwd)),
    }
    k = max(1, int(len(p) * 0.1))
    top = np.argsort(-p)[:k]
    out["top_decile_precision"] = float(y[top].mean()) if k else None
    out["top_decile_mean_fwd_return"] = float(np.nanmean(fwd[top])) if k else None
    return out


class DirectionModel:
    def __init__(self, horizon_days: int, threshold: float = 0.0, seed: int = 42):
        self.horizon_days = horizon_days
        self.threshold = threshold
        self.seed = seed
        self.estimator, self.backend = _make_estimator(seed)
        self.features = list(FEATURE_COLUMNS)
        self.train_start: Optional[str] = None
        self.train_end: Optional[str] = None
        self.n_train: int = 0

    def fit(self, panel: pd.DataFrame) -> "DirectionModel":
        X, y = panel[self.features].values, panel["label"].astype(int).values
        self.estimator.fit(X, y)
        self.train_start = str(panel.index.min().date())
        self.train_end = str(panel.index.max().date())
        self.n_train = len(panel)
        self._importance = self._compute_importance(X, y)
        return self

    def _compute_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Native split-gain importance when the estimator exposes it (LightGBM); otherwise
        permutation importance on a sample of the training rows (scikit-learn fallback)."""
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            from sklearn.inspection import permutation_importance

            n = min(len(X), 3000)
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), n, replace=False) if len(X) > n else np.arange(len(X))
            res = permutation_importance(self.estimator, X[idx], y[idx], n_repeats=3, random_state=self.seed, scoring="roc_auc")
            imp = np.clip(res.importances_mean, 0, None)
        total = float(np.sum(imp)) or 1.0
        return {f: float(v) / total for f, v in sorted(zip(self.features, imp), key=lambda t: -t[1])}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X[self.features].values)[:, 1]

    def feature_importance(self) -> Dict[str, float]:
        return dict(getattr(self, "_importance", {}) or {})


def walk_forward(
    panel: pd.DataFrame,
    horizon_days: int,
    threshold: float = 0.0,
    n_folds: int = 5,
    min_train_rows: int = 500,
    test_months: int = 6,
    seed: int = 42,
) -> WalkForwardReport:
    """Expanding-window walk-forward with a `horizon_days` gap between train and test."""
    dates = pd.DatetimeIndex(sorted(panel.index.unique()))
    if len(dates) < 60:
        raise ValueError("Not enough history for walk-forward evaluation")
    last = dates[-1]
    fold_ends = [last - pd.DateOffset(months=test_months * i) for i in range(n_folds)][::-1]
    folds: List[FoldMetrics] = []
    pooled_y, pooled_p, pooled_fwd = [], [], []
    for i, test_end in enumerate(fold_ends, 1):
        test_start = test_end - pd.DateOffset(months=test_months)
        # Training labels for bar t use bars up to t + horizon; stop training `horizon` trading
        # days before the test window begins so no label peeks into it.
        train_end_pos = dates.searchsorted(test_start) - horizon_days
        if train_end_pos <= 0:
            continue
        train_end = dates[train_end_pos - 1]
        train = panel[panel.index <= train_end]
        test = panel[(panel.index > test_start) & (panel.index <= test_end)]
        if len(train) < min_train_rows or len(test) < 20:
            continue
        model = DirectionModel(horizon_days, threshold, seed).fit(train)
        p = model.predict_proba(test)
        y, fwd = test["label"].astype(int).values, test["fwd_return"].values
        m = _metrics(y, p, fwd)
        folds.append(FoldMetrics(i, str(train_end.date()), str(test_start.date()), str(test_end.date()),
                                 len(train), len(test), **m))
        pooled_y.append(y)
        pooled_p.append(p)
        pooled_fwd.append(fwd)
        logger.info("fold %d: train<=%s test %s..%s acc=%.3f auc=%s", i, train_end.date(), test_start.date(),
                    test_end.date(), m["accuracy"], f"{m['auc']:.3f}" if m["auc"] is not None else "n/a")
    report = WalkForwardReport(horizon_days, threshold, folds)
    if pooled_y:
        report.pooled = _metrics(np.concatenate(pooled_y), np.concatenate(pooled_p), np.concatenate(pooled_fwd))
        report.pooled["n_folds"] = len(folds)
        report.pooled["n_test_rows"] = int(sum(f.n_test for f in folds))
    return report
