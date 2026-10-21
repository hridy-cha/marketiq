"""
Feature and label construction for the direction model.

Features are derived from the indicator frame and expressed as scale-free ratios (so one model can
serve every symbol). The label for bar t is 1 if the adjusted close `horizon` bars later exceeds
close_t * (1 + threshold). Rows whose label would need bars beyond the end of the data are dropped.
`build_dataset` guarantees features at t use only bars <= t and the label uses bars > t.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "rsi_14",
    "macd_hist_pct",
    "macd_pct",
    "price_vs_sma20_pct",
    "price_vs_sma50_pct",
    "price_vs_sma200_pct",
    "sma20_vs_sma50_pct",
    "sma50_vs_sma200_pct",
    "sma_50_slope_pct",
    "atr_pct",
    "realized_vol_20",
    "volume_ratio",
    "obv_vs_sma_pct",
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "bb_position",
    "stretch_atr",
    "rel_ret_20",
    "rel_ret_60",
]


def make_features(ind: pd.DataFrame, bench_ind: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale-free feature frame aligned to `ind.index`."""
    c = ind["close"].astype(float)
    f = pd.DataFrame(index=ind.index)
    f["rsi_14"] = ind["rsi_14"]
    f["macd_hist_pct"] = ind["macd_hist"] / c * 100.0
    f["macd_pct"] = ind["macd"] / c * 100.0
    f["price_vs_sma20_pct"] = (c / ind["sma_20"] - 1.0) * 100.0
    f["price_vs_sma50_pct"] = (c / ind["sma_50"] - 1.0) * 100.0
    f["price_vs_sma200_pct"] = (c / ind["sma_200"] - 1.0) * 100.0
    f["sma20_vs_sma50_pct"] = (ind["sma_20"] / ind["sma_50"] - 1.0) * 100.0
    f["sma50_vs_sma200_pct"] = (ind["sma_50"] / ind["sma_200"] - 1.0) * 100.0
    f["sma_50_slope_pct"] = ind["sma_50_slope_pct"]
    f["atr_pct"] = ind["atr_pct"]
    f["realized_vol_20"] = ind["realized_vol_20"]
    f["volume_ratio"] = ind["volume_ratio"]
    f["obv_vs_sma_pct"] = (ind["obv"] - ind["obv_sma_20"]) / ind["volume_sma_20"].replace(0, np.nan)
    for w in (1, 5, 20, 60):
        f[f"ret_{w}"] = ind[f"ret_{w}"]
    f["pct_from_52w_high"] = ind["pct_from_52w_high"]
    f["pct_from_52w_low"] = ind["pct_from_52w_low"]
    width = (ind["bb_upper"] - ind["bb_lower"]).replace(0, np.nan)
    f["bb_position"] = (c - ind["bb_lower"]) / width
    f["stretch_atr"] = ind["stretch_atr"]
    if bench_ind is not None and not bench_ind.empty:
        b = bench_ind.reindex(ind.index, method="ffill")
        f["rel_ret_20"] = ind["ret_20"] - b["ret_20"]
        f["rel_ret_60"] = ind["ret_60"] - b["ret_60"]
    else:
        f["rel_ret_20"] = np.nan
        f["rel_ret_60"] = np.nan
    return f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def make_labels(ind: pd.DataFrame, horizon: int, threshold: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """Returns (label, forward_return). Both NaN for the last `horizon` bars."""
    base = ind["adj_close"].astype(float) if "adj_close" in ind.columns and ind["adj_close"].notna().any() else ind["close"].astype(float)
    fwd = base.shift(-horizon) / base - 1.0
    label = (fwd > threshold).astype(float)
    label[fwd.isna()] = np.nan
    return label, fwd


def build_dataset(
    indicators: Dict[str, pd.DataFrame],
    horizon: int,
    threshold: float = 0.0,
    bench_ind: Optional[pd.DataFrame] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Stacked panel: columns = FEATURE_COLUMNS + label, fwd_return, symbol; index = (date).
    Drops rows with any missing feature or missing label."""
    frames = []
    for sym, ind in indicators.items():
        if symbols is not None and sym not in symbols:
            continue
        f = make_features(ind, bench_ind)
        label, fwd = make_labels(ind, horizon, threshold)
        f["label"] = label
        f["fwd_return"] = fwd
        f["symbol"] = sym
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["label", "fwd_return", "symbol"])
    panel = pd.concat(frames)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["label"])
    panel.index.name = "date"
    return panel.sort_index(kind="stable")
