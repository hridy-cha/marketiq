"""
Performance metrics for an equity curve.

Formulas adapted from FinRL-X `src/backtest/backtest_engine.py` and
`src/trading/performance_analyzer.py` (Apache 2.0, AI4Finance Foundation): CAGR from total return
over the number of trading days, annualized volatility as daily stdev * sqrt(252), Sharpe as
(annual return - risk free) / annual volatility, Sortino with downside deviation, Calmar as annual
return over max drawdown, max drawdown from the running peak, and daily win rate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    cumulative = values / values.iloc[0]
    running_max = cumulative.cummax()
    return float(((cumulative - running_max) / running_max).min())


def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.02, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n = len(returns)
    annual_return = float((1.0 + total_return) ** (TRADING_DAYS / n) - 1.0) if n > 0 else 0.0
    annual_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
    mdd = max_drawdown(equity)
    out = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0,
        "sortino_ratio": (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar_ratio": annual_return / abs(mdd) if mdd < 0 else 0.0,
        "win_rate": float((returns > 0).mean()) if n else 0.0,
        "trading_days": n,
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
    }
    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.reindex(equity.index).ffill().dropna()
        if len(b) > 1:
            b_total = float(b.iloc[-1] / b.iloc[0] - 1.0)
            b_ret = b.pct_change().dropna()
            aligned = pd.concat([returns, b_ret], axis=1, join="inner").dropna()
            out["benchmark_total_return"] = b_total
            out["benchmark_annual_return"] = float((1.0 + b_total) ** (TRADING_DAYS / max(1, len(b_ret))) - 1.0)
            out["benchmark_max_drawdown"] = max_drawdown(b)
            out["excess_return"] = total_return - b_total
            if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
                beta = float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())
                out["beta"] = beta
    return out
