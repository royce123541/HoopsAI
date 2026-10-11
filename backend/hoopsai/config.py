from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (prefix HOOPSAI_) or .env."""

    model_config = SettingsConfigDict(env_prefix="HOOPSAI_", env_file=".env", extra="ignore")

    env: str = "dev"
    # 127.0.0.1, not localhost: Docker Desktop's IPv6 (::1) port forwarding can hang for 30s
    # per connection before clients fall back to IPv4.
    database_url: str = "postgresql+psycopg://hoopsai:hoopsai@127.0.0.1:5433/hoopsai"
    # Empty or "off": no Redis (serverless hosts); live charts then poll, not WebSockets.
    redis_url: str = "redis://127.0.0.1:6379/0"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_format: Literal["text", "json"] = "text"  # json: one object per line, for containers
    # nba_stats: stats.nba.com (blocks cloud servers); espn: ESPN's public API (works anywhere).
    data_source: Literal["nba_stats", "espn"] = "nba_stats"
    # Play-by-play and raw game payloads, ~140 MB a season, are only needed for in-game
    # retraining; set false on a free 0.5 GB database.
    store_pbp: bool = True
    # Thin finished games' live charts to this many points and keep this many seasons of
    # them (0: keep everything); ~300 MB a season otherwise, too much for a 0.5 GB database.
    live_history_points: int = 0
    live_history_seasons: int = 2
    ingame_retrain: bool = True  # weekly in-game retrain needs ~750 MB; off on small hosts

    @field_validator("redis_url")
    @classmethod
    def _redis_off(cls, url: str) -> str:
        return "" if url.strip().lower() in ("", "off", "none") else url

    @field_validator("database_url")
    @classmethod
    def _use_psycopg(cls, url: str) -> str:
        """Accept plain postgres:// URLs as hosts (e.g. Neon) hand them out."""
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url.removeprefix(prefix)
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
