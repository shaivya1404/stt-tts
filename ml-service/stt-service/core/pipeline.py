"""Speech-to-text pipeline implementation with Faster-Whisper.

This pipeline orchestrates the full STT flow:
1. Audio preprocessing (format conversion, resampling)
2. Optional denoising and echo cancellation
3. Voice activity detection
4. Language identification
5. ASR transcription (Faster-Whisper large-v3)
6. Post-processing (punctuation, truecasing, ITN)
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from common.config import Settings

from .asr_conformer_rnnt import transcribe as conformer_transcribe
from .asr_whisper_fallback import transcribe as whisper_transcribe
from .audio_preprocess import preprocess
from .itn import apply_itn
from .language_id import detect_language
from .language_model import refine_transcript
from .punctuation import add_punctuation
from .quality_scoring import score_quality
from .truecasing import apply_truecase


class SttResult(BaseModel):
    """Result structure for STT transcription."""

    text: str
    language: str
    confidence: float
    timestamps: List[Dict[str, Any]]
    meta: Dict[str, Any] = Field(default_factory=dict)
    modelUsed: str | None = None


class STTPipeline:
    """Orchestrates the STT pipeline with Faster-Whisper backend.

    The pipeline handles:
    - Audio format conversion and resampling to 16kHz mono
    - Language detection (auto or with hint)
    - Primary transcription with fallback mechanism
    - Text post-processing (punctuation, casing, normalization)
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str = "whisper_large-v3",
        model_version: str = "v1",
    ) -> None:
        """Initialize the STT pipeline.

        Args:
            settings: Application settings (device, paths, etc.)
            model_name: Name of the ASR model to use
            model_version: Version of the model
        """
        self.settings = settings
        self.model_name = model_name
        self.model_version = model_version
        self.logger = logger.bind(component="stt_pipeline", model=model_name)
        self.logger.info("STT Pipeline initialized (model=%s, version=%s)", model_name, model_version)

    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> SttResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV, MP3, FLAC, etc.)
            language_hint: Optional language code (e.g., "en", "hi", "ta")
                          If None, language will be auto-detected

        Returns:
            SttResult with transcribed text, confidence, timestamps, and metadata
        """
        if not audio_bytes:
            raise ValueError("audio_bytes payload is empty")

        self.logger.info("Starting STT pipeline (payload=%d bytes, hint=%s)", len(audio_bytes), language_hint)

        # Step 1: Preprocess audio to numpy array
        # This handles format conversion, resampling to 16kHz, and mono conversion
        audio_array, duration_seconds = preprocess(audio_bytes)
        self.logger.debug("Audio preprocessed: %.2f seconds", duration_seconds)

        # Step 2: Detect language if not provided
        # Note: Whisper also detects language, but we do it here for logging
        language = language_hint
        if not language:
            # Use first few seconds of audio for language detection
            language = detect_language(audio_array, language_hint)
            self.logger.debug("Detected language: %s", language)

        # Step 3: Primary ASR transcription
        text, confidence, timestamps, model_used = conformer_transcribe(audio_array, language)
        fallback_used = False

        # Step 4: Fallback to Whisper if confidence is low
        # Note: Currently both use Whisper, so fallback is unlikely to differ
        if confidence < 0.7:
            self.logger.debug("Low confidence (%.2f), trying fallback", confidence)
            fb_text, fb_confidence, fb_timestamps, fb_model = whisper_transcribe(audio_array, language)
            if fb_confidence >= confidence:
                text = fb_text
                confidence = fb_confidence
                timestamps = fb_timestamps
                model_used = fb_model
                fallback_used = True
                self.logger.debug("Using fallback result (confidence=%.2f)", fb_confidence)

        # Step 5: Post-processing
        # These are lightweight text transformations
        refined_text = refine_transcript(text, language)
        punctuated_text = add_punctuation(refined_text, language)
        truecased_text = apply_truecase(punctuated_text, language)
        normalized_text = apply_itn(truecased_text, language)

        # Step 6: Quality scoring
        quality_score = score_quality(normalized_text, confidence)

        # Build metadata
        meta = {
            "duration_seconds": round(duration_seconds, 2),
            "quality_score": round(quality_score, 3),
            "language_hint": language_hint,
            "language_detected": language,
            "fallback_used": fallback_used,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "audio_samples": len(audio_array),
        }

        self.logger.info(
            "STT complete: %d chars, confidence=%.2f, model=%s",
            len(normalized_text),
            confidence,
            model_used,
        )

        return SttResult(
            text=normalized_text,
            language=language,
            confidence=round(confidence, 3),
            timestamps=timestamps,
            meta=meta,
            modelUsed=model_used,
        )
