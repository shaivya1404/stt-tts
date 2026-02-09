"""Audio preprocessing for the STT pipeline.

Converts raw audio bytes to numpy arrays suitable for Faster-Whisper inference.
Handles multiple audio formats and resamples to 16kHz mono as required by Whisper.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf
from loguru import logger

# Whisper models expect 16kHz mono audio
TARGET_SAMPLE_RATE = 16000


def preprocess(audio_bytes: bytes) -> Tuple[np.ndarray, float]:
    """Convert audio bytes to numpy array and return duration.

    Args:
        audio_bytes: Raw audio data in any format supported by soundfile (WAV, FLAC, OGG, etc.)

    Returns:
        Tuple of (audio_array, duration_seconds):
            - audio_array: numpy float32 array normalized to [-1, 1], resampled to 16kHz mono
            - duration_seconds: actual duration of the audio in seconds
    """
    if not audio_bytes:
        raise ValueError("Empty audio bytes provided")

    try:
        # Read audio from bytes buffer
        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

        logger.debug(
            "Loaded audio: shape=%s, sample_rate=%d, dtype=%s",
            audio_data.shape,
            sample_rate,
            audio_data.dtype,
        )

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            logger.debug("Converted stereo to mono")

        # Resample to 16kHz if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            try:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE,
                )
                logger.debug("Resampled from %d Hz to %d Hz", sample_rate, TARGET_SAMPLE_RATE)
            except ImportError:
                logger.warning(
                    "librosa not available for resampling from %d Hz to %d Hz. "
                    "Audio may not be processed correctly.",
                    sample_rate,
                    TARGET_SAMPLE_RATE,
                )

        # Ensure float32
        audio_data = audio_data.astype(np.float32)

        # Calculate duration
        duration_seconds = len(audio_data) / TARGET_SAMPLE_RATE

        logger.debug(
            "Preprocessed audio: %d samples, %.2f seconds",
            len(audio_data),
            duration_seconds,
        )

        return audio_data, duration_seconds

    except Exception as e:
        logger.error("Failed to preprocess audio: %s", str(e))
        raise ValueError(f"Audio preprocessing failed: {str(e)}") from e


def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convenience function to convert audio bytes to numpy array without duration.

    Args:
        audio_bytes: Raw audio data

    Returns:
        Numpy float32 array suitable for Whisper
    """
    audio_array, _ = preprocess(audio_bytes)
    return audio_array
