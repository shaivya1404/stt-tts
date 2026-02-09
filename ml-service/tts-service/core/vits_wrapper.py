"""XTTS v2 text-to-speech synthesis using Coqui TTS.

This module provides high-quality multilingual TTS using the XTTS v2 model.
Supports 17 languages including Hindi and English with voice cloning capabilities.
"""
from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

# Add parent directory to path for common imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings

# Model configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_ID = "xtts_v2:coqui-tts"

# Global model cache to avoid reloading
_tts = None

# Default speaker reference path (will be created if not exists)
DEFAULT_SPEAKER_WAV = None


def _get_tts():
    """Get or initialize the Coqui TTS model.

    Uses lazy loading and caching for performance.
    Model is loaded on first call and reused for subsequent calls.
    """
    global _tts

    if _tts is not None:
        return _tts

    try:
        from TTS.api import TTS

        device = settings.device  # "cuda" or "cpu"

        logger.info(
            "Loading Coqui TTS model: %s (device=%s)",
            MODEL_NAME,
            device,
        )

        _tts = TTS(MODEL_NAME).to(device)

        logger.info("Coqui TTS model loaded successfully")
        return _tts

    except Exception as e:
        logger.error("Failed to load Coqui TTS model: %s", str(e))
        raise RuntimeError(f"TTS model initialization failed: {str(e)}") from e


def _get_default_speaker_wav() -> str | None:
    """Get the default speaker reference WAV file.

    XTTS v2 requires a reference speaker audio for voice cloning.
    Returns None if no default speaker is configured.
    """
    global DEFAULT_SPEAKER_WAV

    if DEFAULT_SPEAKER_WAV and os.path.exists(DEFAULT_SPEAKER_WAV):
        return DEFAULT_SPEAKER_WAV

    # Check for default speaker in models directory
    default_path = Path(settings.model_base_path) / "tts" / "speakers" / "default.wav"
    if default_path.exists():
        DEFAULT_SPEAKER_WAV = str(default_path)
        return DEFAULT_SPEAKER_WAV

    return None


def _map_language_code(language: str) -> str:
    """Map language codes to XTTS v2 supported format.

    XTTS v2 uses specific language codes. This maps common variants.
    """
    language_mapping = {
        "en": "en",
        "en-US": "en",
        "en-GB": "en",
        "en-IN": "en",
        "hi": "hi",
        "hi-IN": "hi",
        "ta": "ta",
        "ta-IN": "ta",
        "te": "te",
        "te-IN": "te",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "pl": "pl",
        "tr": "tr",
        "ru": "ru",
        "nl": "nl",
        "cs": "cs",
        "ar": "ar",
        "zh": "zh-cn",
        "zh-CN": "zh-cn",
        "ja": "ja",
        "ko": "ko",
        "hu": "hu",
    }

    normalized = language.split("-")[0].lower()
    return language_mapping.get(language, language_mapping.get(normalized, "en"))


def synthesize_waveform(
    text: str,
    language: str,
    speaker_wav: str | None = None,
    speed: float = 1.0,
) -> Tuple[bytes, float]:
    """Generate speech audio from text using XTTS v2.

    Args:
        text: Text to synthesize
        language: Language code (e.g., "en", "hi", "ta")
        speaker_wav: Optional path to reference speaker audio for voice cloning
        speed: Speech speed multiplier (0.5 to 2.0)

    Returns:
        Tuple of (audio_bytes, duration_seconds):
            - audio_bytes: WAV file content as bytes
            - duration_seconds: Duration of the generated audio
    """
    logger.debug(
        "Starting XTTS v2 synthesis (text_len=%d, language=%s, speed=%.2f)",
        len(text),
        language,
        speed,
    )

    tts = _get_tts()
    mapped_language = _map_language_code(language)

    # Use provided speaker WAV or try to get default
    reference_wav = speaker_wav or _get_default_speaker_wav()

    try:
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        if reference_wav and os.path.exists(reference_wav):
            # Voice cloning mode with reference speaker
            logger.debug("Using voice cloning with reference: %s", reference_wav)
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=mapped_language,
                speaker_wav=reference_wav,
                speed=speed,
            )
        else:
            # Use model's default speaker if available
            logger.debug("Using default model speaker (no reference provided)")
            # For XTTS v2, we need a speaker reference. Generate a simple one if missing.
            # Check if model supports direct synthesis without speaker_wav
            try:
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=mapped_language,
                    speed=speed,
                )
            except TypeError:
                # Model requires speaker_wav - create a placeholder error
                raise RuntimeError(
                    "XTTS v2 requires a speaker reference WAV file. "
                    "Please provide speaker_wav or configure a default speaker."
                )

        # Read the generated audio
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        # Calculate duration from WAV file
        with wave.open(output_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)

        # Clean up temp file
        os.unlink(output_path)

        logger.info(
            "XTTS v2 synthesis complete: %.2f seconds, %d bytes",
            duration,
            len(audio_bytes),
        )

        return audio_bytes, duration

    except Exception as e:
        # Clean up temp file on error
        if "output_path" in locals() and os.path.exists(output_path):
            os.unlink(output_path)
        logger.error("XTTS v2 synthesis failed: %s", str(e))
        raise RuntimeError(f"TTS synthesis failed: {str(e)}") from e


def synthesize_waveform_legacy(
    phonemes: List[str],
    speaker_embedding: bytes | None,
    style_params: Dict[str, Any],
    language: str,
) -> Tuple[bytes, float]:
    """Legacy interface for backward compatibility with existing pipeline.

    This adapts the old phoneme-based interface to the new text-based XTTS v2.
    Note: Phonemes are ignored as XTTS v2 handles text-to-speech directly.

    Args:
        phonemes: List of phonemes (ignored, kept for API compatibility)
        speaker_embedding: Speaker embedding bytes (used as speaker_wav path if string)
        style_params: Style parameters including speed
        language: Language code

    Returns:
        Tuple of (audio_bytes, duration_seconds)
    """
    # Extract text from style_params if available, otherwise join phonemes
    text = style_params.get("_original_text", " ".join(phonemes))
    speed = style_params.get("speed", 1.0)

    # Speaker embedding might be a path to WAV file
    speaker_wav = None
    if isinstance(speaker_embedding, str) and os.path.exists(speaker_embedding):
        speaker_wav = speaker_embedding

    return synthesize_waveform(text, language, speaker_wav, speed)


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded TTS model.

    Returns:
        Dict with model name, status, and capabilities
    """
    return {
        "name": MODEL_NAME,
        "id": MODEL_ID,
        "loaded": _tts is not None,
        "device": settings.device,
        "languages": ["en", "hi", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"],
        "features": ["voice_cloning", "multilingual", "speed_control"],
    }
