"""Typed loaders for the YAML configuration files (universe, scoring weights, signal rules)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from marketiq.config.settings import get_settings


class UniverseTicker(BaseModel):
    symbol: str
    name: str = ""
    sector: str = ""


class Universe(BaseModel):
    benchmark: str = "SPY"
    tickers: List[UniverseTicker]

    @property
    def symbols(self) -> List[str]:
        return [t.symbol for t in self.tickers]

    @property
    def all_symbols(self) -> List[str]:
        """Universe plus benchmark, deduplicated, order preserved."""
        seen = []
        for s in self.symbols + [self.benchmark]:
            if s not in seen:
                seen.append(s)
        return seen


class TrendConfig(BaseModel):
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200


class MomentumConfig(BaseModel):
    rsi_period: int = 14
    rsi_ideal_low: float = 50
    rsi_ideal_high: float = 70
    return_windows: List[int] = Field(default_factory=lambda: [20, 60])


class VolumeConfig(BaseModel):
    volume_sma: int = 20


class VolatilityConfig(BaseModel):
    atr_period: int = 14
    atr_pct_good: float = 1.5
    atr_pct_bad: float = 5.0


class FundamentalsConfig(BaseModel):
    factors: Dict[str, int] = Field(default_factory=dict)
    max_staleness_days: int = 400


class SentimentConfig(BaseModel):
    lookback_days: int = 14
    half_life_days: float = 5
    min_articles: int = 2


class MLConfig(BaseModel):
    model_name: str = "direction_lgbm"
    min_walk_forward_auc: float = 0.53
    min_walk_forward_balanced_accuracy: float = 0.52


class ScoringConfig(BaseModel):
    weights: Dict[str, float]
    trend: TrendConfig = Field(default_factory=TrendConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    volume: VolumeConfig = Field(default_factory=VolumeConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    fundamentals: FundamentalsConfig = Field(default_factory=FundamentalsConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    KNOWN_COMPONENTS: ClassVar[tuple] = ("trend", "momentum", "volume", "volatility", "fundamentals", "sentiment", "ml")

    @model_validator(mode="after")
    def _validate_weights(self) -> "ScoringConfig":
        unknown = set(self.weights) - set(self.KNOWN_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown scoring components: {sorted(unknown)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Scoring weights must be non-negative")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("At least one scoring weight must be positive")
        self.weights = {k: v / total for k, v in self.weights.items()}
        return self


class SignalThresholds(BaseModel):
    buy_score: float = 70
    sell_score: float = 35
    watch_low: float = 55
    min_history_days: int = 210


class RiskFlagConfig(BaseModel):
    earnings_within_days: int = 7
    max_stretch_atr: float = 3.0
    rsi_overbought: float = 80
    rsi_oversold: float = 20
    max_atr_pct: float = 6.0


class BuyRule(BaseModel):
    require_trend_bullish: bool = True
    require_no_sell_patterns: bool = True
    block_if_extended: bool = True
    block_if_rsi_overbought: bool = True


class SellRule(BaseModel):
    any_of: List[str] = Field(default_factory=list)


class WatchRule(BaseModel):
    catalysts: List[str] = Field(default_factory=list)


class SignalRules(BaseModel):
    buy: BuyRule = Field(default_factory=BuyRule)
    sell: SellRule = Field(default_factory=SellRule)
    watch: WatchRule = Field(default_factory=WatchRule)


class SignalConfig(BaseModel):
    thresholds: SignalThresholds = Field(default_factory=SignalThresholds)
    risk_flags: RiskFlagConfig = Field(default_factory=RiskFlagConfig)
    rules: SignalRules = Field(default_factory=SignalRules)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_universe(path: Optional[Path] = None) -> Universe:
    path = path or get_settings().data.path("universe_file")
    return Universe(**_read_yaml(path))


def load_scoring_config(path: Optional[Path] = None) -> ScoringConfig:
    path = path or get_settings().data.path("scoring_file")
    return ScoringConfig(**_read_yaml(path))


def load_signal_config(path: Optional[Path] = None) -> SignalConfig:
    path = path or get_settings().data.path("signals_file")
    return SignalConfig(**_read_yaml(path))


@lru_cache(maxsize=1)
def cached_universe() -> Universe:
    return load_universe()


def clear_config_caches() -> None:
    cached_universe.cache_clear()
