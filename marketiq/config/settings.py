"""
Application settings.

Adapted from FinRL-X `src/config/settings.py` (Apache 2.0, AI4Finance Foundation): same
pydantic-settings pattern with env prefixes, reorganized for MarketIQ. Secrets are SecretStr and are
never logged. Live trading is disabled by default and requires an explicit opt-in.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class DatabaseSettings(BaseSettings):
    """SQLite by default. Set DATABASE_URL=postgresql+psycopg://user:pass@host/db to use Postgres."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", extra="ignore")
    url: str = "sqlite:///data/marketiq.db"
    echo: bool = False

    def resolved_url(self) -> str:
        if self.url.startswith("sqlite:///") and not self.url.startswith("sqlite:////"):
            rel = self.url[len("sqlite:///") :]
            return f"sqlite:///{_resolve(rel)}"
        return self.url


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")
    base_dir: str = "data"
    universe_file: str = "config/universe.yaml"
    scoring_file: str = "config/scoring.yaml"
    signals_file: str = "config/signals.yaml"
    models_dir: str = "models"
    finrlx_db: str = "data/finrl_trading.db"
    # Preferred live price/fundamentals provider: "yfinance" (free, default) or "fmp".
    price_provider: str = "yfinance"
    news_provider: str = "yfinance"
    exchange: str = "NYSE"

    def path(self, attr: str) -> Path:
        return _resolve(getattr(self, attr))


class FMPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FMP_", extra="ignore")
    api_key: Optional[SecretStr] = None
    base_url: str = "https://financialmodelingprep.com/stable"


class FinnhubSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINNHUB_", extra="ignore")
    api_key: Optional[SecretStr] = None


class OpenAISettings(BaseSettings):
    """Optional LLM sentiment. The default sentiment analyzer is an offline lexicon."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")
    api_key: Optional[SecretStr] = None
    model: str = "gpt-4o-mini"
    request_timeout: int = 30


class AlpacaSettings(BaseSettings):
    """Optional broker adapter. Paper endpoint only unless ENABLE_LIVE_TRADING=true."""

    model_config = SettingsConfigDict(env_prefix="APCA_", extra="ignore")
    api_key: Optional[SecretStr] = None
    api_secret: Optional[SecretStr] = None
    base_url: str = "https://paper-api.alpaca.markets"

    def is_paper_url(self) -> bool:
        return "paper-api" in self.base_url


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore")
    horizon_days: int = 20
    return_threshold: float = 0.0
    min_train_rows: int = 500


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MarketIQ"
    environment: str = "development"
    log_level: str = "INFO"
    # Hard safety default. Real-money trading is refused unless this is explicitly true.
    enable_live_trading: bool = False

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    fmp: FMPSettings = Field(default_factory=FMPSettings)
    finnhub: FinnhubSettings = Field(default_factory=FinnhubSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("environment")
    @classmethod
    def _env(cls, v: str) -> str:
        if v not in {"development", "testing", "staging", "production"}:
            raise ValueError("environment must be development, testing, staging or production")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


def override_for_tests(**env: str) -> None:
    """Set environment variables and clear the settings cache. Used by test fixtures."""
    for k, v in env.items():
        os.environ[k] = v
    get_settings.cache_clear()
