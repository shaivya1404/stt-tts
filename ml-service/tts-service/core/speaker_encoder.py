"""Speaker encoder using Resemblyzer for d-vector extraction.

Provides speaker embedding extraction for:
- Voice cloning
- Speaker verification
- Multi-speaker TTS control

Dependencies:
    pip install resemblyzer
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

# Try to import resemblyzer
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RESEMBLYZER_SR
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    RESEMBLYZER_SR = 16000
    logger.warning("resemblyzer not available. Install with: pip install resemblyzer")

# Cached encoder
_voice_encoder: Optional["VoiceEncoder"] = None

# Speaker embedding cache
_embedding_cache: Dict[str, np.ndarray] = {}


def get_voice_encoder() -> Optional["VoiceEncoder"]:
    """Get cached voice encoder.

    Returns:
        VoiceEncoder instance or None if not available
    """
    global _voice_encoder

    if not RESEMBLYZER_AVAILABLE:
        return None

    if _voice_encoder is None:
        logger.info("Loading Resemblyzer voice encoder...")
        try:
            _voice_encoder = VoiceEncoder()
            logger.info("Voice encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load voice encoder: {e}")
            return None

    return _voice_encoder


def load_audio_for_embedding(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
) -> Optional[np.ndarray]:
    """Load and preprocess audio for embedding extraction.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Preprocessed audio array or None
    """
    try:
        import soundfile as sf

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        # Load audio
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                logger.warning(f"librosa not available for resampling from {sr} to {target_sr}")
                return None

        return audio

    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None


def load_audio_bytes_for_embedding(
    audio_bytes: bytes,
    target_sr: int = 16000,
) -> Optional[np.ndarray]:
    """Load audio bytes for embedding extraction.

    Args:
        audio_bytes: Raw audio bytes
        target_sr: Target sample rate

    Returns:
        Preprocessed audio array or None
    """
    try:
        import soundfile as sf

        # Read from bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                logger.warning("librosa not available for resampling")
                return None

        return audio

    except Exception as e:
        logger.error(f"Failed to load audio bytes: {e}")
        return None


def extract_embedding(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> Optional[np.ndarray]:
    """Extract speaker embedding from audio.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Speaker embedding (256-dimensional) or None
    """
    encoder = get_voice_encoder()
    if encoder is None:
        return None

    try:
        # Preprocess if needed (Resemblyzer expects specific format)
        if RESEMBLYZER_AVAILABLE:
            # Preprocess for voice encoder
            processed = preprocess_wav(audio)
            if len(processed) < 1600:  # Minimum length for embedding
                logger.warning("Audio too short for embedding extraction")
                return None

            # Extract embedding
            embedding = encoder.embed_utterance(processed)
            return embedding
        else:
            return None

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


def extract_embedding_from_file(
    audio_path: Union[str, Path],
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """Extract speaker embedding from audio file.

    Args:
        audio_path: Path to audio file
        use_cache: Whether to use cached embeddings

    Returns:
        Speaker embedding or None
    """
    audio_path = str(audio_path)

    # Check cache
    if use_cache and audio_path in _embedding_cache:
        logger.debug(f"Using cached embedding for {audio_path}")
        return _embedding_cache[audio_path]

    # Load audio
    audio = load_audio_for_embedding(audio_path)
    if audio is None:
        return None

    # Extract embedding
    embedding = extract_embedding(audio)
    if embedding is not None and use_cache:
        _embedding_cache[audio_path] = embedding

    return embedding


def extract_embedding_from_bytes(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Extract speaker embedding from audio bytes.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Speaker embedding or None
    """
    audio = load_audio_bytes_for_embedding(audio_bytes)
    if audio is None:
        return None

    return extract_embedding(audio)


def compute_speaker_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """Compute cosine similarity between speaker embeddings.

    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, (similarity + 1) / 2)))


def get_speaker_embedding(
    voice_id: Optional[str] = None,
    audio_sample_path: Optional[str] = None,
) -> Optional[bytes]:
    """Get speaker embedding for TTS.

    Args:
        voice_id: Voice identifier (for cached/registered speakers)
        audio_sample_path: Path to reference audio sample

    Returns:
        Speaker embedding as bytes or None
    """
    if not voice_id and not audio_sample_path:
        logger.debug("No voice reference provided; using default speaker")
        return None

    embedding = None

    # Try to get from audio sample
    if audio_sample_path:
        logger.debug(f"Extracting embedding from audio sample: {audio_sample_path}")
        embedding = extract_embedding_from_file(audio_sample_path)

    # Try to get from voice_id cache
    if embedding is None and voice_id:
        # Check if voice_id corresponds to a cached embedding
        cache_key = f"voice:{voice_id}"
        if cache_key in _embedding_cache:
            embedding = _embedding_cache[cache_key]
            logger.debug(f"Using cached embedding for voice_id: {voice_id}")

    if embedding is not None:
        # Convert to bytes for storage/transmission
        return embedding.tobytes()

    logger.warning(f"Could not get embedding for voice_id={voice_id}, sample={audio_sample_path}")
    return None


def register_speaker(
    voice_id: str,
    audio_paths: List[Union[str, Path]],
) -> Optional[np.ndarray]:
    """Register a speaker with multiple audio samples.

    Creates an averaged embedding from multiple samples for better quality.

    Args:
        voice_id: Unique voice identifier
        audio_paths: List of reference audio paths

    Returns:
        Averaged speaker embedding or None
    """
    if not audio_paths:
        logger.error("No audio paths provided for speaker registration")
        return None

    embeddings = []
    for path in audio_paths:
        embedding = extract_embedding_from_file(path, use_cache=False)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        logger.error(f"Failed to extract any embeddings for voice_id: {voice_id}")
        return None

    # Average embeddings
    averaged = np.mean(embeddings, axis=0)

    # Normalize
    averaged = averaged / np.linalg.norm(averaged)

    # Cache with voice_id
    cache_key = f"voice:{voice_id}"
    _embedding_cache[cache_key] = averaged

    logger.info(f"Registered speaker {voice_id} with {len(embeddings)} samples")
    return averaged


def embedding_to_tensor(embedding: Union[bytes, np.ndarray]) -> "torch.Tensor":
    """Convert embedding to PyTorch tensor.

    Args:
        embedding: Speaker embedding (bytes or numpy array)

    Returns:
        PyTorch tensor
    """
    import torch

    if isinstance(embedding, bytes):
        embedding = np.frombuffer(embedding, dtype=np.float32)

    return torch.from_numpy(embedding).float()


def clear_cache() -> None:
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
    logger.info("Speaker embedding cache cleared")
