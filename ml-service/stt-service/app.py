"""FastAPI application for the modular STT service."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import (  # noqa: E402  # pylint: disable=wrong-import-position
    ModelInfo,
    configure_logging,
    get_active_models,
    register_model,
    set_model_status,
    settings,
)
from core import STTPipeline


class TimestampSegment(BaseModel):
    start: float
    end: float
    word: str | None = None


class SttTranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float
    timestamps: List[TimestampSegment]
    meta: Dict[str, Any] = Field(default_factory=dict)
    modelUsed: str | None = None
    status: str = "success"


class StatusResponse(BaseModel):
    status: str
    detail: str
    models: List[Dict[str, Any]] = Field(default_factory=list)


MODEL_NAME = "whisper_large-v3"
MODEL_VERSION = "v1"
pipeline: STTPipeline | None = None
app = FastAPI(title="STT Service", version="0.4.0")


def _model_path() -> str:
    return f"{settings.model_base_path}/stt/{MODEL_NAME}/{MODEL_VERSION}"


def _register_default_model(status: Literal["loading", "ready", "error"] = "ready") -> ModelInfo:
    model_info = ModelInfo(
        name=MODEL_NAME,
        type="stt",
        version=MODEL_VERSION,
        status=status,
        path=_model_path(),
        config={"device": settings.device},
    )
    return register_model(model_info)


async def _initialize_pipeline() -> None:
    global pipeline
    pipeline = STTPipeline(settings=settings, model_name=MODEL_NAME, model_version=MODEL_VERSION)


@app.on_event("startup")
async def startup_event() -> None:
    configure_logging(settings.log_level)
    _register_default_model(status="loading")
    await _initialize_pipeline()
    set_model_status("stt", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    logger.info("STT service started in %s mode", settings.environment)


@app.get("/ml/stt/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    models = [model.model_dump() for model in get_active_models("stt")]
    return StatusResponse(status="ok", detail="stt-service healthy", models=models)


@app.get("/ml/stt/models")
async def list_models() -> Dict[str, Any]:
    return {"models": [model.model_dump() for model in get_active_models("stt")]}


@app.post("/ml/stt/initialize", response_model=StatusResponse)
async def initialize_models() -> StatusResponse:
    set_model_status("stt", MODEL_NAME, MODEL_VERSION, "loading")
    await _initialize_pipeline()
    set_model_status("stt", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    models = [model.model_dump() for model in get_active_models("stt")]
    return StatusResponse(status="ok", detail="STT pipeline reinitialized", models=models)


@app.post("/ml/stt/reload", response_model=StatusResponse)
async def reload_models() -> StatusResponse:
    set_model_status("stt", MODEL_NAME, MODEL_VERSION, "loading")
    await asyncio.sleep(0.1)
    await _initialize_pipeline()
    set_model_status("stt", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    models = [model.model_dump() for model in get_active_models("stt")]
    return StatusResponse(status="ok", detail="STT models reloaded", models=models)


@app.post("/ml/stt/transcribe", response_model=SttTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language_hint: str | None = Form(default=None),
) -> SttTranscribeResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="STT pipeline not initialized")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    result = pipeline.transcribe(payload, language_hint)
    timestamps = [TimestampSegment(**segment) for segment in result.timestamps]
    return SttTranscribeResponse(
        text=result.text,
        language=result.language,
        confidence=result.confidence,
        timestamps=timestamps,
        meta=result.meta,
        modelUsed=result.modelUsed,
    )


@app.post("/ml/stt/stream", status_code=501)
async def stream_stub() -> Dict[str, Any]:
    return {
        "status": "not_implemented",
        "detail": "Real-time streaming will be delivered via WebSockets/GRPC in a future phase",
    }
