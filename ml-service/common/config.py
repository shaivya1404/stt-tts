"""Common configuration utilities for ML services."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _autodetect_device() -> Literal["cpu", "cuda"]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch is optional
        pass
    return "cpu"


class Settings(BaseSettings):
    """Application settings shared across ML services."""

    model_base_path: str = Field(default="/models", alias="MODEL_BASE_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    device: str = Field(default_factory=_autodetect_device, alias="DEVICE")

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
