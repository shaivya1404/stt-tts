"""Quality scoring placeholder.

TODO: Integrate confidence calibration / MOSNet for ASR outputs.
"""
from __future__ import annotations

from loguru import logger


def score_quality(text: str, confidence: float) -> float:
    """Return a pseudo QA score."""
    score = min(1.0, max(0.0, confidence + 0.1)) if text else 0.0
    logger.debug("Computed quality score {}", score)
    return score
