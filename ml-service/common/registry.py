"""Simple in-memory model registry shared by ML services."""
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field

ModelType = Literal["tts", "stt"]
ModelStatus = Literal["loading", "ready", "error"]


class ModelInfo(BaseModel):
    name: str
    type: ModelType
    version: str
    status: ModelStatus = "loading"
    path: str
    config: Dict[str, Any] = Field(default_factory=dict)


_REGISTRY: Dict[Tuple[str, str, str], ModelInfo] = {}
_REGISTRY_LOCK = Lock()


def _registry_key(model_type: ModelType, name: str, version: str) -> Tuple[str, str, str]:
    return (model_type, name, version)


def register_model(model_info: ModelInfo) -> ModelInfo:
    with _REGISTRY_LOCK:
        _REGISTRY[_registry_key(model_info.type, model_info.name, model_info.version)] = model_info
        return model_info


def get_active_models(model_type: ModelType | None = None) -> List[ModelInfo]:
    with _REGISTRY_LOCK:
        values = list(_REGISTRY.values())
    if model_type:
        return [model for model in values if model.type == model_type]
    return values


def set_model_status(
    model_type: ModelType,
    name: str,
    version: str,
    status: ModelStatus,
    *,
    path: str | None = None,
    config: Dict[str, Any] | None = None,
) -> ModelInfo:
    with _REGISTRY_LOCK:
        key = _registry_key(model_type, name, version)
        existing = _REGISTRY.get(key)
        if existing is None:
            existing = ModelInfo(
                name=name,
                type=model_type,
                version=version,
                status=status,
                path=path or "",
                config=config or {},
            )
        else:
            update_data: Dict[str, Any] = {"status": status}
            if path is not None:
                update_data["path"] = path
            if config is not None:
                update_data["config"] = config
            existing = existing.model_copy(update=update_data)
        _REGISTRY[key] = existing
        return existing
