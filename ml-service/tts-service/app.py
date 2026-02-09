from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
from core import TTSPipeline, TtsRequest, TtsResult  # noqa: E402


class StatusResponse(BaseModel):
    status: str
    detail: str
    service: str = Field(default="tts-service")
    models: List[Dict[str, Any]] = Field(default_factory=list)


MODEL_NAME = "xtts_v2"
MODEL_VERSION = "v1"
pipeline: TTSPipeline | None = None
app = FastAPI(title="TTS Service", version="0.4.0")


def _model_path() -> str:
    return f"{settings.model_base_path}/tts/{MODEL_NAME}/{MODEL_VERSION}"


def _register_default_model(status: Literal["loading", "ready", "error"] = "ready") -> ModelInfo:
    model_info = ModelInfo(
        name=MODEL_NAME,
        type="tts",
        version=MODEL_VERSION,
        status=status,
        path=_model_path(),
        config={"device": settings.device},
    )
    return register_model(model_info)


async def _initialize_pipeline() -> None:
    global pipeline
    pipeline = TTSPipeline(settings=settings, model_name=MODEL_NAME, model_version=MODEL_VERSION)


@app.on_event("startup")
async def startup_event() -> None:
    configure_logging(settings.log_level)
    _register_default_model(status="loading")
    await _initialize_pipeline()
    set_model_status("tts", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    logger.info("TTS service started in {} mode", settings.environment)


@app.get("/ml/tts/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    models = [model.model_dump() for model in get_active_models("tts")]
    return StatusResponse(status="ok", detail="tts-service healthy", models=models)


@app.get("/ml/tts/models")
async def list_models() -> Dict[str, Any]:
    return {"models": [model.model_dump() for model in get_active_models("tts")]}


@app.post("/ml/tts/initialize", response_model=StatusResponse)
async def initialize_models() -> StatusResponse:
    set_model_status("tts", MODEL_NAME, MODEL_VERSION, "loading")
    await _initialize_pipeline()
    set_model_status("tts", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    models = [model.model_dump() for model in get_active_models("tts")]
    return StatusResponse(status="ok", detail="TTS pipeline reinitialized", models=models)


@app.post("/ml/tts/reload", response_model=StatusResponse)
async def reload_models() -> StatusResponse:
    set_model_status("tts", MODEL_NAME, MODEL_VERSION, "loading")
    await asyncio.sleep(0.1)
    await _initialize_pipeline()
    set_model_status("tts", MODEL_NAME, MODEL_VERSION, "ready", path=_model_path())
    models = [model.model_dump() for model in get_active_models("tts")]
    return StatusResponse(status="ok", detail="TTS models reloaded", models=models)


@app.post("/ml/tts/predict", response_model=TtsResult)
async def synthesize_speech(payload: TtsRequest) -> TtsResult:
    """Synthesize speech from text.

    Returns audio file path, download URL, and base64-encoded audio data.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TTS pipeline not initialized")

    result = pipeline.synthesize(payload)
    if not result.audio_path:
        raise HTTPException(status_code=500, detail="Failed to synthesize audio")

    # Add download URL
    filename = Path(result.audio_path).name
    result.audio_url = f"/ml/tts/audio/{filename}"

    # Add base64 encoded audio for remote access
    try:
        with open(result.audio_path, "rb") as f:
            result.audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to encode audio as base64: %s", e)

    return result


@app.get("/ml/tts/audio/{filename}")
async def download_audio(filename: str) -> FileResponse:
    """Download a generated audio file by filename.

    Args:
        filename: The audio filename (e.g., 'default_abc123.wav')

    Returns:
        The audio file as a downloadable response
    """
    # Validate filename to prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Search for the file in possible output directories
    possible_paths = [
        Path(settings.model_base_path) / "tts" / MODEL_NAME / MODEL_VERSION / "synthesized" / filename,
        Path("/tmp/models") / "tts" / MODEL_NAME / MODEL_VERSION / "synthesized" / filename,
        Path("/outputs") / filename,  # For custom deployments
    ]

    for audio_path in possible_paths:
        if audio_path.exists():
            return FileResponse(
                path=str(audio_path),
                media_type="audio/wav",
                filename=filename,
            )

    raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
