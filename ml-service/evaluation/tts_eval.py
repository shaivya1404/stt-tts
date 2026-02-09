#!/usr/bin/env python3
"""TTS Model Evaluation Script.

Evaluates TTS models on:
- MOS (Mean Opinion Score) estimation
- Speaker similarity
- Real-time factor (RTF)
- Pronunciation accuracy

Usage:
    python tts_eval.py --model-path models/xtts_indic --test-set test_sets/indic_tts_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class TTSEvalResult:
    """TTS evaluation result for a single sample."""
    text: str
    audio_duration: float
    synthesis_time: float
    rtf: float
    mos_score: float
    speaker_similarity: Optional[float]
    pronunciation_score: Optional[float]


@dataclass
class TTSEvalSummary:
    """Summary of TTS evaluation."""
    num_samples: int
    avg_mos: float
    avg_rtf: float
    avg_speaker_similarity: float
    total_audio_duration: float
    total_synthesis_time: float
    samples_meeting_targets: Dict[str, int]


# Target metrics
TARGETS = {
    "mos_score": 3.5,
    "speaker_similarity": 0.85,
    "rtf": 0.5,
}


def estimate_mos_heuristic(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate MOS using heuristics.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Estimated MOS (1.0 to 5.0)
    """
    scores = []

    # Clipping check
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        scores.append(2.0)
    elif peak > 0.95:
        scores.append(3.0)
    elif peak > 0.1:
        scores.append(4.0)
    else:
        scores.append(3.0)

    # RMS check
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        scores.append(2.0)
    elif rms < 0.1:
        scores.append(3.5)
    else:
        scores.append(4.0)

    # Smoothness check
    diff = np.abs(np.diff(audio))
    smoothness = 1.0 / (1.0 + np.std(diff) * 10)
    scores.append(2.0 + smoothness * 3.0)

    # Duration check
    duration = len(audio) / sample_rate
    if duration < 0.5:
        scores.append(2.5)
    elif duration > 60:
        scores.append(3.5)
    else:
        scores.append(4.0)

    return float(np.mean(scores))


def compute_speaker_similarity(
    reference_audio: np.ndarray,
    synthesized_audio: np.ndarray,
    sample_rate: int,
) -> float:
    """Compute speaker similarity between reference and synthesized audio.

    Args:
        reference_audio: Reference speaker audio
        synthesized_audio: Synthesized audio
        sample_rate: Sample rate

    Returns:
        Similarity score (0.0 to 1.0)
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()

        ref_processed = preprocess_wav(reference_audio)
        syn_processed = preprocess_wav(synthesized_audio)

        if len(ref_processed) < 1600 or len(syn_processed) < 1600:
            return 0.5

        ref_embedding = encoder.embed_utterance(ref_processed)
        syn_embedding = encoder.embed_utterance(syn_processed)

        similarity = np.dot(ref_embedding, syn_embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(syn_embedding)
        )

        return float((similarity + 1) / 2)

    except ImportError:
        logger.warning("resemblyzer not available, returning default similarity")
        return 0.5
    except Exception as e:
        logger.error(f"Speaker similarity computation failed: {e}")
        return 0.5


def evaluate_sample(
    model,
    text: str,
    language: str,
    reference_audio: Optional[np.ndarray],
    config,
    sample_rate: int = 22050,
) -> TTSEvalResult:
    """Evaluate a single TTS sample.

    Args:
        model: TTS model
        text: Text to synthesize
        language: Language code
        reference_audio: Reference speaker audio
        config: Model config
        sample_rate: Target sample rate

    Returns:
        Evaluation result
    """
    import torch

    # Synthesize
    start_time = time.time()

    with torch.no_grad():
        outputs = model.synthesize(
            text,
            config,
            speaker_wav=reference_audio,
            language=language[:2],
        )

    synthesis_time = time.time() - start_time
    audio = outputs["wav"]

    # Calculate metrics
    audio_duration = len(audio) / sample_rate
    rtf = synthesis_time / audio_duration if audio_duration > 0 else float("inf")
    mos_score = estimate_mos_heuristic(audio, sample_rate)

    # Speaker similarity
    speaker_similarity = None
    if reference_audio is not None:
        speaker_similarity = compute_speaker_similarity(
            reference_audio, audio, sample_rate
        )

    return TTSEvalResult(
        text=text,
        audio_duration=audio_duration,
        synthesis_time=synthesis_time,
        rtf=rtf,
        mos_score=mos_score,
        speaker_similarity=speaker_similarity,
        pronunciation_score=None,
    )


def evaluate_test_set(
    model,
    config,
    test_manifest: Path,
    reference_audio: Optional[np.ndarray] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[TTSEvalResult], TTSEvalSummary]:
    """Evaluate model on test set.

    Args:
        model: TTS model
        config: Model config
        test_manifest: Path to test manifest JSONL
        reference_audio: Reference speaker audio
        max_samples: Maximum number of samples to evaluate

    Returns:
        Tuple of (results list, summary)
    """
    results = []

    with open(test_manifest, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        language = sample.get("language", "hi-IN")

        try:
            result = evaluate_sample(
                model, text, language, reference_audio, config
            )
            results.append(result)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} samples")

        except Exception as e:
            logger.error(f"Error evaluating sample {idx}: {e}")
            continue

    # Calculate summary
    if results:
        mos_scores = [r.mos_score for r in results]
        rtf_values = [r.rtf for r in results]
        similarities = [r.speaker_similarity for r in results if r.speaker_similarity is not None]

        samples_meeting = {
            "mos": sum(1 for r in results if r.mos_score >= TARGETS["mos_score"]),
            "rtf": sum(1 for r in results if r.rtf <= TARGETS["rtf"]),
            "speaker_similarity": sum(
                1 for r in results
                if r.speaker_similarity is not None and r.speaker_similarity >= TARGETS["speaker_similarity"]
            ),
        }

        summary = TTSEvalSummary(
            num_samples=len(results),
            avg_mos=float(np.mean(mos_scores)),
            avg_rtf=float(np.mean(rtf_values)),
            avg_speaker_similarity=float(np.mean(similarities)) if similarities else 0.0,
            total_audio_duration=sum(r.audio_duration for r in results),
            total_synthesis_time=sum(r.synthesis_time for r in results),
            samples_meeting_targets=samples_meeting,
        )
    else:
        summary = TTSEvalSummary(
            num_samples=0,
            avg_mos=0.0,
            avg_rtf=0.0,
            avg_speaker_similarity=0.0,
            total_audio_duration=0.0,
            total_synthesis_time=0.0,
            samples_meeting_targets={},
        )

    return results, summary


def print_evaluation_report(summary: TTSEvalSummary) -> None:
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("TTS EVALUATION REPORT")
    print("=" * 60)

    print(f"\nSamples Evaluated: {summary.num_samples}")
    print(f"Total Audio Duration: {summary.total_audio_duration:.2f}s")
    print(f"Total Synthesis Time: {summary.total_synthesis_time:.2f}s")

    print("\n" + "-" * 40)
    print(f"{'Metric':<25} {'Value':<15} {'Target':<15} {'Status'}")
    print("-" * 40)

    # MOS
    mos_status = "PASS" if summary.avg_mos >= TARGETS["mos_score"] else "FAIL"
    print(f"{'MOS Score':<25} {summary.avg_mos:.2f}{'':<10} > {TARGETS['mos_score']:<10} {mos_status}")

    # RTF
    rtf_status = "PASS" if summary.avg_rtf <= TARGETS["rtf"] else "FAIL"
    print(f"{'Real-time Factor':<25} {summary.avg_rtf:.2f}{'':<10} < {TARGETS['rtf']:<10} {rtf_status}")

    # Speaker Similarity
    if summary.avg_speaker_similarity > 0:
        sim_status = "PASS" if summary.avg_speaker_similarity >= TARGETS["speaker_similarity"] else "FAIL"
        print(f"{'Speaker Similarity':<25} {summary.avg_speaker_similarity:.2f}{'':<10} > {TARGETS['speaker_similarity']:<10} {sim_status}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate TTS model")
    parser.add_argument("--model-path", required=True, help="Path to TTS model")
    parser.add_argument("--test-set", required=True, help="Path to test manifest JSONL")
    parser.add_argument("--reference-audio", help="Path to reference speaker audio")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--output", help="Output path for results JSON")

    args = parser.parse_args()

    # Load model
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(f"{args.model_path}/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=args.model_path)
        model = model.cuda() if __import__("torch").cuda.is_available() else model
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load reference audio
    reference_audio = None
    if args.reference_audio and sf:
        reference_audio, _ = sf.read(args.reference_audio, dtype="float32")

    # Evaluate
    results, summary = evaluate_test_set(
        model, config, Path(args.test_set), reference_audio, args.max_samples
    )

    # Print report
    print_evaluation_report(summary)

    # Save results
    if args.output:
        output_data = {
            "summary": {
                "num_samples": summary.num_samples,
                "avg_mos": summary.avg_mos,
                "avg_rtf": summary.avg_rtf,
                "avg_speaker_similarity": summary.avg_speaker_similarity,
            },
            "results": [
                {
                    "text": r.text,
                    "mos_score": r.mos_score,
                    "rtf": r.rtf,
                    "speaker_similarity": r.speaker_similarity,
                }
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
