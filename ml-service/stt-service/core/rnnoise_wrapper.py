"""Neural noise reduction using DeepFilterNet.

Provides speech enhancement and denoising for:
- Pre-processing noisy audio before ASR
- Real-time noise reduction
- Post-processing enhancement

Dependencies:
    pip install deepfilternet
"""
from __future__ import annotations

import io
from typing import Optional, Tuple, Union

import numpy as np
from loguru import logger

# Try to import DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df import config as df_config
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False
    logger.warning("DeepFilterNet not available. Install with: pip install deepfilternet")

# Try soundfile for audio I/O
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Cached DeepFilterNet model
_df_model = None
_df_state = None


def get_deepfilter_model():
    """Get cached DeepFilterNet model.

    Returns:
        Tuple of (model, state) or (None, None)
    """
    global _df_model, _df_state

    if not DEEPFILTER_AVAILABLE:
        return None, None

    if _df_model is None:
        logger.info("Loading DeepFilterNet model...")
        try:
            _df_model, _df_state, _ = init_df()
            logger.info("DeepFilterNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepFilterNet: {e}")
            return None, None

    return _df_model, _df_state


def denoise_deepfilter(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Denoise audio using DeepFilterNet.

    Args:
        audio: Audio array (mono, float32)
        sample_rate: Sample rate

    Returns:
        Denoised audio array
    """
    model, state = get_deepfilter_model()
    if model is None:
        return audio

    try:
        import torch

        # DeepFilterNet expects 48kHz audio
        target_sr = 48000

        # Resample if needed
        if sample_rate != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            except ImportError:
                logger.warning(f"librosa not available for resampling from {sample_rate} to {target_sr}")
                return audio

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        # Enhance
        enhanced = enhance(model, state, audio_tensor)

        # Convert back to numpy
        enhanced_np = enhanced.squeeze().numpy()

        # Resample back if needed
        if sample_rate != target_sr:
            try:
                import librosa
                enhanced_np = librosa.resample(enhanced_np, orig_sr=target_sr, target_sr=sample_rate)
            except ImportError:
                pass

        return enhanced_np

    except Exception as e:
        logger.error(f"DeepFilterNet enhancement failed: {e}")
        return audio


def denoise_spectral_subtraction(
    audio: np.ndarray,
    sample_rate: int,
    noise_reduce_factor: float = 0.9,
) -> np.ndarray:
    """Simple spectral subtraction noise reduction.

    Fallback when DeepFilterNet is not available.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        noise_reduce_factor: Noise reduction strength (0-1)

    Returns:
        Denoised audio
    """
    try:
        # Simple noise gate approach
        # Estimate noise floor from quietest parts
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop

        # Calculate frame energies
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energies = []
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)

        energies = np.array(energies)

        # Estimate noise floor (lowest 10% of energies)
        noise_floor = np.percentile(energies, 10)

        # Apply soft noise gate
        result = audio.copy()
        for i in range(num_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            frame = audio[start:end]
            energy = np.sqrt(np.mean(frame ** 2))

            if energy < noise_floor * 2:
                # Reduce volume for quiet frames
                gain = max(0.1, 1.0 - noise_reduce_factor * (1.0 - energy / (noise_floor * 2)))
                result[start:end] = frame * gain

        return result

    except Exception as e:
        logger.error(f"Spectral subtraction failed: {e}")
        return audio


def denoise(audio_bytes: bytes) -> bytes:
    """Denoise audio bytes.

    Uses DeepFilterNet if available, falls back to spectral subtraction.

    Args:
        audio_bytes: Input audio bytes (WAV format)

    Returns:
        Denoised audio bytes
    """
    logger.debug(f"Denoising audio ({len(audio_bytes)} bytes)")

    if not SOUNDFILE_AVAILABLE:
        logger.warning("soundfile not available, returning original audio")
        return audio_bytes

    try:
        # Load audio
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Apply denoising
        if DEEPFILTER_AVAILABLE:
            denoised = denoise_deepfilter(audio, sr)
        else:
            denoised = denoise_spectral_subtraction(audio, sr)

        # Convert back to bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, denoised, sr, format='WAV')
        output_buffer.seek(0)

        logger.debug("Audio denoising complete")
        return output_buffer.read()

    except Exception as e:
        logger.error(f"Denoising failed: {e}")
        return audio_bytes


def denoise_array(
    audio: np.ndarray,
    sample_rate: int,
    use_neural: bool = True,
) -> np.ndarray:
    """Denoise audio array.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        use_neural: Whether to use neural denoising

    Returns:
        Denoised audio array
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if use_neural and DEEPFILTER_AVAILABLE:
        return denoise_deepfilter(audio, sample_rate)
    else:
        return denoise_spectral_subtraction(audio, sample_rate)


def estimate_snr(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate Signal-to-Noise Ratio.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Estimated SNR in dB
    """
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    if len(audio) < frame_length:
        return 0.0

    num_frames = (len(audio) - frame_length) // hop_length + 1
    energies = []

    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        energy = np.sum(frame ** 2) / frame_length
        energies.append(energy)

    energies = np.array(energies)

    if len(energies) < 2:
        return 0.0

    # Assume lower 20% of energies are noise
    sorted_energies = np.sort(energies)
    noise_threshold = max(1, int(len(sorted_energies) * 0.2))

    noise_energy = np.mean(sorted_energies[:noise_threshold])
    signal_energy = np.mean(sorted_energies[noise_threshold:])

    if noise_energy <= 0:
        return 50.0

    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


def should_denoise(audio: np.ndarray, sample_rate: int, snr_threshold: float = 15.0) -> bool:
    """Determine if audio should be denoised.

    Args:
        audio: Audio array
        sample_rate: Sample rate
        snr_threshold: SNR threshold below which denoising is recommended

    Returns:
        True if denoising is recommended
    """
    snr = estimate_snr(audio, sample_rate)
    return snr < snr_threshold


class StreamingDenoiser:
    """Streaming denoiser for real-time applications."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size_ms: int = 20,
        use_neural: bool = True,
    ):
        """Initialize streaming denoiser.

        Args:
            sample_rate: Audio sample rate
            frame_size_ms: Frame size in milliseconds
            use_neural: Whether to use neural denoising
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.use_neural = use_neural and DEEPFILTER_AVAILABLE
        self.buffer = np.array([], dtype=np.float32)

        if self.use_neural:
            self.model, self.state = get_deepfilter_model()

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process

        Returns:
            Denoised audio chunk
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk.flatten()])

        # Process complete frames
        if len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            self.buffer = self.buffer[self.frame_size:]

            if self.use_neural and self.model is not None:
                return denoise_deepfilter(frame, self.sample_rate)
            else:
                return denoise_spectral_subtraction(frame, self.sample_rate)

        return np.array([], dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Flush remaining buffer.

        Returns:
            Remaining processed audio
        """
        if len(self.buffer) > 0:
            remaining = self.buffer
            self.buffer = np.array([], dtype=np.float32)
            return remaining
        return np.array([], dtype=np.float32)

    def reset(self):
        """Reset the denoiser state."""
        self.buffer = np.array([], dtype=np.float32)
