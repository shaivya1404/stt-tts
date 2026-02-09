"""Voice Activity Detection using Silero VAD.

Provides accurate, lightweight VAD for:
- Speech segment detection
- Pre-processing audio before ASR
- Filtering silence and noise

Dependencies:
    Silero VAD is loaded via torch.hub (no separate install needed)
    pip install torch torchaudio
"""
from __future__ import annotations

import io
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger

# Silero VAD model cache
_vad_model = None
_vad_utils = None


def get_vad_model():
    """Load Silero VAD model (cached).

    Returns:
        Tuple of (model, utils)
    """
    global _vad_model, _vad_utils

    if _vad_model is None:
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD model loaded successfully")

    return _vad_model, _vad_utils


def audio_bytes_to_tensor(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """Convert audio bytes to torch tensor.

    Args:
        audio_bytes: Raw audio bytes (WAV format)
        sample_rate: Expected sample rate

    Returns:
        Tuple of (audio tensor, actual sample rate)
    """
    try:
        import soundfile as sf

        # Read audio from bytes
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if needed
        if sr != sample_rate:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate
            except ImportError:
                logger.warning(f"librosa not available for resampling from {sr} to {sample_rate}")

        return torch.from_numpy(audio_data).float(), sr

    except Exception as e:
        logger.error(f"Failed to convert audio bytes to tensor: {e}")
        raise


def detect_speech_segments(
    audio_bytes: bytes,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    sample_rate: int = 16000,
    return_seconds: bool = True,
) -> List[Tuple[float, float]]:
    """Detect speech segments in audio using Silero VAD.

    Args:
        audio_bytes: Raw audio bytes (WAV format)
        threshold: VAD threshold (0.0 to 1.0). Higher = more aggressive filtering
        min_speech_duration_ms: Minimum speech duration to keep
        min_silence_duration_ms: Minimum silence duration to split segments
        window_size_samples: Window size for VAD (512 for 16kHz)
        sample_rate: Audio sample rate (must be 16kHz for Silero)
        return_seconds: If True, return timestamps in seconds; else in samples

    Returns:
        List of (start, end) tuples for speech segments
    """
    logger.debug(f"Running VAD on {len(audio_bytes)} bytes")

    try:
        # Load model
        model, utils = get_vad_model()
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

        # Convert audio
        audio_tensor, sr = audio_bytes_to_tensor(audio_bytes, sample_rate)

        # Silero VAD requires 16kHz
        if sr != 16000:
            logger.warning(f"Silero VAD requires 16kHz audio, got {sr}Hz")
            # Try to resample
            try:
                import torchaudio.functional as F
                audio_tensor = F.resample(audio_tensor.unsqueeze(0), sr, 16000).squeeze(0)
            except Exception:
                logger.warning("Could not resample, using original sample rate")

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            return_seconds=return_seconds,
            sampling_rate=16000,
        )

        # Convert to list of tuples
        segments = [(ts['start'], ts['end']) for ts in speech_timestamps]

        logger.debug(f"VAD found {len(segments)} speech segments")
        return segments

    except Exception as e:
        logger.error(f"VAD failed: {e}")
        # Fallback: return whole audio as single segment
        duration = len(audio_bytes) / (sample_rate * 2)  # Assuming 16-bit audio
        return [(0.0, max(duration, 1.0))]


def detect_speech_segments_streaming(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> List[Tuple[float, float]]:
    """Streaming-compatible VAD for real-time applications.

    Args:
        audio_tensor: Audio as torch tensor
        sample_rate: Audio sample rate
        threshold: VAD threshold
        min_speech_duration_ms: Minimum speech duration
        min_silence_duration_ms: Minimum silence duration

    Returns:
        List of (start, end) tuples in seconds
    """
    model, utils = get_vad_model()
    (get_speech_timestamps, _, _, VADIterator, _) = utils

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
        sampling_rate=sample_rate,
    )

    return [(ts['start'], ts['end']) for ts in speech_timestamps]


def get_speech_probability(
    audio_chunk: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
) -> float:
    """Get speech probability for a single audio chunk.

    Args:
        audio_chunk: Audio chunk (1D array/tensor)
        sample_rate: Sample rate

    Returns:
        Speech probability (0.0 to 1.0)
    """
    model, _ = get_vad_model()

    if isinstance(audio_chunk, np.ndarray):
        audio_chunk = torch.from_numpy(audio_chunk).float()

    # Silero expects 16kHz
    if sample_rate != 16000:
        try:
            import torchaudio.functional as F
            audio_chunk = F.resample(audio_chunk.unsqueeze(0), sample_rate, 16000).squeeze(0)
        except Exception:
            pass

    with torch.no_grad():
        speech_prob = model(audio_chunk, 16000).item()

    return speech_prob


def filter_audio_by_vad(
    audio_bytes: bytes,
    threshold: float = 0.5,
    sample_rate: int = 16000,
    pad_ms: int = 30,
) -> bytes:
    """Filter audio to keep only speech segments.

    Args:
        audio_bytes: Input audio bytes
        threshold: VAD threshold
        sample_rate: Sample rate
        pad_ms: Padding around speech segments in ms

    Returns:
        Audio bytes with only speech segments
    """
    import soundfile as sf

    # Get speech segments
    segments = detect_speech_segments(
        audio_bytes,
        threshold=threshold,
        sample_rate=sample_rate,
        return_seconds=False,
    )

    if not segments:
        logger.warning("No speech detected in audio")
        return audio_bytes

    # Convert audio
    audio_tensor, sr = audio_bytes_to_tensor(audio_bytes, sample_rate)
    audio_np = audio_tensor.numpy()

    # Collect speech chunks
    pad_samples = int(pad_ms * sr / 1000)
    speech_chunks = []

    for start, end in segments:
        start_padded = max(0, int(start) - pad_samples)
        end_padded = min(len(audio_np), int(end) + pad_samples)
        speech_chunks.append(audio_np[start_padded:end_padded])

    # Concatenate chunks
    if speech_chunks:
        filtered_audio = np.concatenate(speech_chunks)
    else:
        filtered_audio = audio_np

    # Convert back to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, filtered_audio, sr, format='WAV')
    output_buffer.seek(0)

    return output_buffer.read()


class VADIteratorWrapper:
    """Wrapper for streaming VAD with state management."""

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """Initialize VAD iterator.

        Args:
            threshold: VAD threshold
            sample_rate: Audio sample rate
            min_silence_duration_ms: Minimum silence to end segment
            speech_pad_ms: Padding around speech
        """
        model, utils = get_vad_model()
        _, _, _, VADIterator, _ = utils

        self.vad_iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.sample_rate = sample_rate

    def process_chunk(self, audio_chunk: Union[np.ndarray, torch.Tensor]) -> Optional[dict]:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process

        Returns:
            Dict with 'start' or 'end' key if speech boundary detected, else None
        """
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float()

        return self.vad_iterator(audio_chunk)

    def reset(self):
        """Reset VAD state."""
        self.vad_iterator.reset_states()
