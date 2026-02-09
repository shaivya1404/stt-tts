"""Faster-Whisper ASR implementation using large-v3 model.

This module provides high-accuracy speech-to-text transcription using the
Faster-Whisper library (CTranslate2-optimized Whisper implementation).
Supports 99 languages including Hindi, Tamil, Telugu, Bengali, and English.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger

# Add parent directory to path for common imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings

# Model configuration
MODEL_NAME = "large-v3"
MODEL_ID = f"whisper_{MODEL_NAME}:faster-whisper"

# Global model cache to avoid reloading
_model = None


def _get_model():
    """Get or initialize the Faster-Whisper model.

    Uses lazy loading and caching for performance.
    Model is loaded on first call and reused for subsequent calls.
    """
    global _model

    if _model is not None:
        return _model

    try:
        from faster_whisper import WhisperModel

        device = settings.device  # "cuda" or "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading Faster-Whisper model: %s (device=%s, compute_type=%s)",
            MODEL_NAME,
            device,
            compute_type,
        )

        _model = WhisperModel(
            MODEL_NAME,
            device=device,
            compute_type=compute_type,
            download_root=f"{settings.model_base_path}/whisper",
        )

        logger.info("Faster-Whisper model loaded successfully")
        return _model

    except Exception as e:
        logger.error("Failed to load Faster-Whisper model: %s", str(e))
        raise RuntimeError(f"Whisper model initialization failed: {str(e)}") from e


def transcribe(
    audio_data: np.ndarray,
    language: str | None,
) -> Tuple[str, float, List[Dict[str, Any]], str]:
    """Transcribe audio using Faster-Whisper large-v3 model.

    Args:
        audio_data: Numpy array of audio samples (float32, 16kHz mono)
        language: Optional language hint (e.g., "en", "hi", "ta")
                  If None, language will be auto-detected

    Returns:
        Tuple of (text, confidence, timestamps, model_used):
            - text: Full transcribed text
            - confidence: Average confidence score (0.0 to 1.0)
            - timestamps: List of word-level timestamps with format:
                [{"start": float, "end": float, "word": str}, ...]
            - model_used: Model identifier string
    """
    logger.debug(
        "Starting Faster-Whisper transcription (samples=%d, language=%s)",
        len(audio_data),
        language,
    )

    model = _get_model()

    try:
        # Transcribe with word-level timestamps
        segments, info = model.transcribe(
            audio_data,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
            beam_size=5,
            best_of=5,
        )

        # Process segments and extract text, confidence, timestamps
        full_text_parts = []
        all_timestamps = []
        total_confidence = 0.0
        segment_count = 0

        for segment in segments:
            full_text_parts.append(segment.text.strip())
            segment_count += 1

            # Calculate segment confidence from avg_logprob
            # avg_logprob is log probability, convert to 0-1 scale
            segment_confidence = min(1.0, max(0.0, 1.0 + (segment.avg_logprob / 5.0)))
            total_confidence += segment_confidence

            # Extract word-level timestamps if available
            if segment.words:
                for word in segment.words:
                    all_timestamps.append({
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "word": word.word.strip(),
                    })
            else:
                # Fallback to segment-level timestamp
                all_timestamps.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "word": segment.text.strip(),
                })

        # Combine results
        full_text = " ".join(full_text_parts).strip()
        avg_confidence = total_confidence / max(segment_count, 1)

        # Use detected language if not provided
        detected_language = info.language if hasattr(info, "language") else (language or "en")

        logger.info(
            "Transcription complete: %d chars, confidence=%.2f, language=%s",
            len(full_text),
            avg_confidence,
            detected_language,
        )

        return full_text, avg_confidence, all_timestamps, MODEL_ID

    except Exception as e:
        logger.error("Faster-Whisper transcription failed: %s", str(e))
        raise RuntimeError(f"Transcription failed: {str(e)}") from e


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model.

    Returns:
        Dict with model name, status, and capabilities
    """
    return {
        "name": MODEL_NAME,
        "id": MODEL_ID,
        "loaded": _model is not None,
        "device": settings.device,
        "languages": ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"],
        "features": ["word_timestamps", "vad_filter", "language_detection"],
    }
