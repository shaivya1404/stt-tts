"""TTS pipeline using XTTS v2 for high-quality multilingual synthesis.

This pipeline orchestrates the full TTS flow:
1. Language detection
2. Text normalization
3. Speech synthesis (XTTS v2)
4. Audio post-processing
5. Quality estimation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from common import Settings

from . import (
    audio_postprocess,
    language_id,
    quality_mosnet,
    text_normalization,
    vits_wrapper,
    vocoder_hifigan,
)


class TtsRequest(BaseModel):
    """Schema describing incoming synthesis payloads."""

    text: str = Field(..., min_length=1)
    language: str | None = None
    voice_id: str | None = None
    emotion: str | None = None
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    speaker_wav: str | None = None  # Path to reference speaker audio for voice cloning


class TtsResult(BaseModel):
    """Structured response emitted by the pipeline."""

    audio_path: str | None = None
    audio_url: str | None = None  # Relative URL for downloading audio
    audio_base64: str | None = None  # Base64 encoded audio for direct access
    duration: float | None = None
    status: str = "success"
    meta: Dict[str, Any] = Field(default_factory=dict)


class TTSPipeline:
    """Orchestrates the XTTS v2 TTS pipeline.

    The pipeline handles:
    - Language detection (auto or with hint)
    - Text normalization for the target language
    - Speech synthesis with optional voice cloning
    - Audio post-processing and quality estimation
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str = "xtts_v2",
        model_version: str = "v1",
    ) -> None:
        """Initialize the TTS pipeline.

        Args:
            settings: Application settings (device, paths, etc.)
            model_name: Name of the TTS model to use
            model_version: Version of the model
        """
        self.settings = settings
        self.model_name = model_name
        self.model_version = model_version
        self.logger = logger.bind(component="tts_pipeline", model=model_name, version=model_version)
        self.logger.info("TTS Pipeline initialized (model=%s, version=%s)", model_name, model_version)

    def synthesize(self, request: TtsRequest) -> TtsResult:
        """Synthesize speech from text.

        Args:
            request: TTS request with text, language, voice, and style options

        Returns:
            TtsResult with audio file path, duration, and metadata
        """
        self.logger.info(
            "Starting synthesis (chars=%d, voice=%s, language=%s)",
            len(request.text),
            request.voice_id or "default",
            request.language or "auto",
        )

        # Step 1: Detect language if not provided
        detected_language = language_id.detect_language(request.text, request.language)
        self.logger.debug("Language: %s", detected_language)

        # Step 2: Normalize text for the target language
        normalized_text = text_normalization.normalize_text(request.text, detected_language)
        self.logger.debug("Normalized text: %d chars", len(normalized_text))

        # Step 3: Synthesize speech using XTTS v2
        # Pass the original text directly - XTTS v2 handles text-to-speech end-to-end
        audio_bytes, duration = vits_wrapper.synthesize_waveform(
            text=normalized_text,
            language=detected_language,
            speaker_wav=request.speaker_wav,
            speed=request.speed,
        )
        self.logger.debug("Synthesis complete: %.2f seconds", duration)

        # Step 4: Apply vocoder (pass-through for XTTS v2)
        vocoded_audio = vocoder_hifigan.vocode(audio_bytes, detected_language)

        # Step 5: Post-process audio (denoising, enhancement)
        enhanced_audio = audio_postprocess.denoise_and_enhance(vocoded_audio)

        # Step 6: Estimate audio quality
        mos_score = quality_mosnet.estimate_mos(enhanced_audio)
        self.logger.debug("Quality score (MOS): %.2f", mos_score)

        # Step 7: Save audio to file
        filename = f"{request.voice_id or 'default'}_{uuid4().hex}.wav"
        output_dir = Path(self.settings.model_base_path) / "tts" / self.model_name / self.model_version / "synthesized"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback to temp directory if model path is not writable
            fallback_dir = Path("/tmp/models") / "tts" / self.model_name / self.model_version / "synthesized"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            output_dir = fallback_dir
            self.logger.warning("Using fallback output directory: %s", output_dir)

        output_path = output_dir / filename

        # Write the audio bytes to the file
        with open(output_path, "wb") as f:
            f.write(enhanced_audio)

        self.logger.info("Audio saved to: %s", output_path)

        # Build metadata
        meta = {
            "language": detected_language,
            "language_requested": request.language,
            "voice_id": request.voice_id,
            "speed": request.speed,
            "mos_score": round(mos_score, 3),
            "model": self.model_name,
            "version": self.model_version,
            "char_count": len(request.text),
            "normalized_char_count": len(normalized_text),
            "audio_size_bytes": len(enhanced_audio),
        }

        self.logger.info(
            "Completed synthesis -> %s (duration=%.2fs, mos=%.2f)",
            output_path,
            duration,
            mos_score,
        )

        return TtsResult(
            audio_path=str(output_path),
            duration=round(duration, 4),
            status="success",
            meta=meta,
        )
