"""Audio quality estimation using MOSNet and related models.

Provides Mean Opinion Score (MOS) prediction for:
- TTS output quality assessment
- Synthesis quality monitoring
- A/B testing of different TTS configurations

Dependencies:
    pip install speechmos
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from loguru import logger

# Try to import speechmos
try:
    import speechmos
    from speechmos import Scorer as MOSScorer
    SPEECHMOS_AVAILABLE = True
except ImportError:
    SPEECHMOS_AVAILABLE = False
    logger.warning("speechmos not available. Install with: pip install speechmos")

# Try alternative: UTMOS
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Cached scorer
_mos_scorer: Optional["MOSScorer"] = None
_utmos_model = None


def get_mos_scorer() -> Optional["MOSScorer"]:
    """Get cached MOS scorer.

    Returns:
        MOSScorer instance or None if not available
    """
    global _mos_scorer

    if not SPEECHMOS_AVAILABLE:
        return None

    if _mos_scorer is None:
        logger.info("Loading MOSNet scorer...")
        try:
            _mos_scorer = MOSScorer()
            logger.info("MOSNet scorer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MOSNet scorer: {e}")
            return None

    return _mos_scorer


def load_audio(
    audio_source: Union[str, Path, bytes],
    target_sr: int = 16000,
) -> Tuple[Optional[np.ndarray], int]:
    """Load audio from file path or bytes.

    Args:
        audio_source: File path or audio bytes
        target_sr: Target sample rate

    Returns:
        Tuple of (audio array, sample rate)
    """
    try:
        import soundfile as sf

        if isinstance(audio_source, bytes):
            audio, sr = sf.read(io.BytesIO(audio_source), dtype='float32')
        else:
            audio, sr = sf.read(str(audio_source), dtype='float32')

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                pass

        return audio, sr

    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None, 0


def estimate_mos_speechmos(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Estimate MOS using speechmos library.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Estimated MOS (1.0 to 5.0)
    """
    scorer = get_mos_scorer()
    if scorer is None:
        return 3.0  # Default mid-range score

    try:
        # speechmos expects audio at 16kHz
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                pass

        # Get MOS score
        score = scorer.score(audio)
        return float(score)

    except Exception as e:
        logger.error(f"MOS estimation failed: {e}")
        return 3.0


def estimate_mos_heuristic(audio_bytes: bytes) -> float:
    """Estimate MOS using heuristics.

    Fallback when neural models are not available.

    Args:
        audio_bytes: WAV audio bytes

    Returns:
        Estimated MOS (1.0 to 5.0)
    """
    if not audio_bytes:
        logger.warning("Empty audio bytes received")
        return 1.0

    audio, sr = load_audio(audio_bytes)
    if audio is None:
        return 2.0

    # Heuristic-based quality estimation
    scores = []

    # 1. Check for clipping
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        clipping_score = 2.0
    elif peak > 0.95:
        clipping_score = 3.0
    elif peak > 0.1:
        clipping_score = 4.0
    else:
        clipping_score = 3.0  # Too quiet
    scores.append(clipping_score)

    # 2. Check signal-to-noise ratio (rough estimate)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        snr_score = 2.0  # Too quiet
    elif rms < 0.1:
        snr_score = 3.5
    else:
        snr_score = 4.0
    scores.append(snr_score)

    # 3. Check for spectral continuity (rough smoothness check)
    try:
        diff = np.abs(np.diff(audio))
        smoothness = 1.0 / (1.0 + np.std(diff) * 10)
        smoothness_score = 2.0 + smoothness * 3.0
        scores.append(smoothness_score)
    except Exception:
        scores.append(3.0)

    # 4. Check duration (very short or very long may indicate issues)
    duration = len(audio) / sr if sr > 0 else 0
    if duration < 0.5:
        duration_score = 2.5
    elif duration > 60:
        duration_score = 3.5
    else:
        duration_score = 4.0
    scores.append(duration_score)

    # Average scores
    final_score = np.mean(scores)
    return float(min(5.0, max(1.0, final_score)))


def estimate_mos(audio_bytes: bytes) -> float:
    """Estimate MOS score for synthesized audio.

    Uses neural model if available, falls back to heuristics.

    Args:
        audio_bytes: WAV audio bytes from synthesis

    Returns:
        Estimated MOS score (1.0 to 5.0 scale)
    """
    if not audio_bytes:
        logger.warning("Empty audio bytes received for quality estimation")
        return 1.0

    # Try neural model first
    if SPEECHMOS_AVAILABLE:
        audio, sr = load_audio(audio_bytes)
        if audio is not None:
            score = estimate_mos_speechmos(audio, sr)
            logger.debug(f"MOSNet quality score: {score:.2f}")
            return score

    # Fallback to heuristics
    score = estimate_mos_heuristic(audio_bytes)
    logger.debug(f"Heuristic quality score: {score:.2f}")
    return score


def estimate_mos_from_file(audio_path: Union[str, Path]) -> float:
    """Estimate MOS from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Estimated MOS score
    """
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return estimate_mos(audio_bytes)
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        return 2.0


def get_quality_metrics(audio_bytes: bytes) -> Dict[str, float]:
    """Get comprehensive quality metrics for audio.

    Args:
        audio_bytes: WAV audio bytes

    Returns:
        Dictionary of quality metrics
    """
    audio, sr = load_audio(audio_bytes)

    if audio is None:
        return {
            "mos": 1.0,
            "peak_amplitude": 0.0,
            "rms": 0.0,
            "duration": 0.0,
            "sample_rate": 0,
        }

    # Basic metrics
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    duration = len(audio) / sr

    # MOS score
    mos = estimate_mos(audio_bytes)

    # Crest factor (peak-to-RMS ratio)
    crest_factor = peak / rms if rms > 0 else 0.0

    # Dynamic range (rough estimate)
    percentile_high = np.percentile(np.abs(audio), 95)
    percentile_low = np.percentile(np.abs(audio), 5)
    dynamic_range = 20 * np.log10(percentile_high / percentile_low) if percentile_low > 0 else 0

    return {
        "mos": mos,
        "peak_amplitude": peak,
        "rms": rms,
        "duration": duration,
        "sample_rate": sr,
        "crest_factor": float(crest_factor),
        "dynamic_range_db": float(dynamic_range),
    }


def is_quality_acceptable(
    audio_bytes: bytes,
    min_mos: float = 3.0,
    min_duration: float = 0.5,
    max_peak: float = 0.99,
) -> Tuple[bool, str]:
    """Check if audio quality meets minimum requirements.

    Args:
        audio_bytes: WAV audio bytes
        min_mos: Minimum acceptable MOS
        min_duration: Minimum duration in seconds
        max_peak: Maximum peak amplitude (clipping threshold)

    Returns:
        Tuple of (is_acceptable, reason)
    """
    metrics = get_quality_metrics(audio_bytes)

    if metrics["mos"] < min_mos:
        return False, f"MOS too low: {metrics['mos']:.2f} < {min_mos}"

    if metrics["duration"] < min_duration:
        return False, f"Duration too short: {metrics['duration']:.2f}s < {min_duration}s"

    if metrics["peak_amplitude"] > max_peak:
        return False, f"Audio clipping detected: peak={metrics['peak_amplitude']:.3f}"

    if metrics["rms"] < 0.01:
        return False, f"Audio too quiet: RMS={metrics['rms']:.4f}"

    return True, "Quality acceptable"
