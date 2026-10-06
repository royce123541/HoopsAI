from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (prefix HOOPSAI_) or .env."""

    model_config = SettingsConfigDict(env_prefix="HOOPSAI_", env_file=".env", extra="ignore")

    env: str = "dev"
    # 127.0.0.1, not localhost: Docker Desktop's IPv6 (::1) port forwarding can hang for 30s
    # per connection before clients fall back to IPv4.
    database_url: str = "postgresql+psycopg://hoopsai:hoopsai@127.0.0.1:5433/hoopsai"
    redis_url: str = "redis://127.0.0.1:6379/0"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_format: Literal["text", "json"] = "text"  # json: one object per line, for containers
    ingame_retrain: bool = True  # weekly in-game retrain needs ~750 MB; off on small hosts


@lru_cache
def get_settings() -> Settings:
    return Settings()
