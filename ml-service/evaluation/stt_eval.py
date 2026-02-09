#!/usr/bin/env python3
"""STT Model Evaluation Script.

Evaluates STT models on:
- WER (Word Error Rate)
- CER (Character Error Rate)
- Processing time

Usage:
    python stt_eval.py --model-path models/whisper_indic --test-set test_sets/indic_stt_test.jsonl
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
class STTEvalResult:
    """STT evaluation result for a single sample."""
    audio_path: str
    reference: str
    prediction: str
    wer: float
    cer: float
    audio_duration: float
    processing_time: float
    rtf: float


@dataclass
class STTEvalSummary:
    """Summary of STT evaluation."""
    num_samples: int
    avg_wer: float
    avg_cer: float
    total_wer: float
    total_cer: float
    avg_rtf: float
    total_audio_duration: float
    total_processing_time: float
    samples_meeting_targets: Dict[str, int]


# Target metrics
TARGETS = {
    "wer_clean": 0.15,
    "wer_noisy": 0.25,
    "cer_clean": 0.10,
    "cer_noisy": 0.15,
}


def compute_wer(reference: str, prediction: str) -> float:
    """Compute Word Error Rate.

    Args:
        reference: Reference transcript
        prediction: Predicted transcript

    Returns:
        WER value
    """
    try:
        from jiwer import wer
        return wer(reference, prediction)
    except ImportError:
        # Fallback implementation
        ref_words = reference.lower().split()
        pred_words = prediction.lower().split()

        if len(ref_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0

        # Simple Levenshtein at word level
        d = np.zeros((len(ref_words) + 1, len(pred_words) + 1))
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(pred_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(pred_words) + 1):
                if ref_words[i - 1] == pred_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,      # deletion
                        d[i][j - 1] + 1,      # insertion
                        d[i - 1][j - 1] + 1,  # substitution
                    )

        return d[len(ref_words)][len(pred_words)] / len(ref_words)


def compute_cer(reference: str, prediction: str) -> float:
    """Compute Character Error Rate.

    Args:
        reference: Reference transcript
        prediction: Predicted transcript

    Returns:
        CER value
    """
    try:
        from jiwer import cer
        return cer(reference, prediction)
    except ImportError:
        # Fallback implementation
        ref_chars = list(reference.lower().replace(" ", ""))
        pred_chars = list(prediction.lower().replace(" ", ""))

        if len(ref_chars) == 0:
            return 1.0 if len(pred_chars) > 0 else 0.0

        d = np.zeros((len(ref_chars) + 1, len(pred_chars) + 1))
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(pred_chars) + 1):
            d[0][j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(pred_chars) + 1):
                if ref_chars[i - 1] == pred_chars[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,
                        d[i][j - 1] + 1,
                        d[i - 1][j - 1] + 1,
                    )

        return d[len(ref_chars)][len(pred_chars)] / len(ref_chars)


def transcribe_audio(
    model,
    processor,
    audio_path: str,
    device: str = "cuda",
) -> Tuple[str, float, float]:
    """Transcribe audio file.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        device: Device to use

    Returns:
        Tuple of (transcription, audio_duration, processing_time)
    """
    import torch

    # Load audio
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            pass

    audio_duration = len(audio) / 16000

    # Process
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)

    # Transcribe
    start_time = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=225)
    processing_time = time.time() - start_time

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription, audio_duration, processing_time


def evaluate_sample(
    model,
    processor,
    audio_path: str,
    reference: str,
    device: str = "cuda",
) -> STTEvalResult:
    """Evaluate a single STT sample.

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        reference: Reference transcript
        device: Device to use

    Returns:
        Evaluation result
    """
    prediction, audio_duration, processing_time = transcribe_audio(
        model, processor, audio_path, device
    )

    wer = compute_wer(reference, prediction)
    cer = compute_cer(reference, prediction)
    rtf = processing_time / audio_duration if audio_duration > 0 else float("inf")

    return STTEvalResult(
        audio_path=audio_path,
        reference=reference,
        prediction=prediction,
        wer=wer,
        cer=cer,
        audio_duration=audio_duration,
        processing_time=processing_time,
        rtf=rtf,
    )


def evaluate_test_set(
    model,
    processor,
    test_manifest: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Tuple[List[STTEvalResult], STTEvalSummary]:
    """Evaluate model on test set.

    Args:
        model: Whisper model
        processor: Whisper processor
        test_manifest: Path to test manifest JSONL
        device: Device to use
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
        audio_path = sample.get("audio_filepath", "")
        reference = sample.get("text", "")

        if not audio_path or not reference:
            continue

        try:
            result = evaluate_sample(model, processor, audio_path, reference, device)
            results.append(result)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(samples)} samples")

        except Exception as e:
            logger.error(f"Error evaluating sample {idx}: {e}")
            continue

    # Calculate summary
    if results:
        wer_values = [r.wer for r in results]
        cer_values = [r.cer for r in results]
        rtf_values = [r.rtf for r in results]

        # Calculate total WER/CER (weighted by reference length)
        total_ref_words = sum(len(r.reference.split()) for r in results)
        total_ref_chars = sum(len(r.reference) for r in results)

        total_word_errors = sum(r.wer * len(r.reference.split()) for r in results)
        total_char_errors = sum(r.cer * len(r.reference) for r in results)

        samples_meeting = {
            "wer_clean": sum(1 for r in results if r.wer <= TARGETS["wer_clean"]),
            "cer_clean": sum(1 for r in results if r.cer <= TARGETS["cer_clean"]),
        }

        summary = STTEvalSummary(
            num_samples=len(results),
            avg_wer=float(np.mean(wer_values)),
            avg_cer=float(np.mean(cer_values)),
            total_wer=total_word_errors / total_ref_words if total_ref_words > 0 else 0,
            total_cer=total_char_errors / total_ref_chars if total_ref_chars > 0 else 0,
            avg_rtf=float(np.mean(rtf_values)),
            total_audio_duration=sum(r.audio_duration for r in results),
            total_processing_time=sum(r.processing_time for r in results),
            samples_meeting_targets=samples_meeting,
        )
    else:
        summary = STTEvalSummary(
            num_samples=0,
            avg_wer=0.0,
            avg_cer=0.0,
            total_wer=0.0,
            total_cer=0.0,
            avg_rtf=0.0,
            total_audio_duration=0.0,
            total_processing_time=0.0,
            samples_meeting_targets={},
        )

    return results, summary


def print_evaluation_report(summary: STTEvalSummary) -> None:
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("STT EVALUATION REPORT")
    print("=" * 60)

    print(f"\nSamples Evaluated: {summary.num_samples}")
    print(f"Total Audio Duration: {summary.total_audio_duration:.2f}s ({summary.total_audio_duration/3600:.2f} hrs)")
    print(f"Total Processing Time: {summary.total_processing_time:.2f}s")
    print(f"Average Real-time Factor: {summary.avg_rtf:.2f}x")

    print("\n" + "-" * 50)
    print(f"{'Metric':<25} {'Value':<15} {'Target':<15} {'Status'}")
    print("-" * 50)

    # WER
    wer_status = "PASS" if summary.total_wer <= TARGETS["wer_clean"] else "FAIL"
    print(f"{'WER (Total)':<25} {summary.total_wer:.1%}{'':<10} < {TARGETS['wer_clean']:.0%}{'':<10} {wer_status}")

    # CER
    cer_status = "PASS" if summary.total_cer <= TARGETS["cer_clean"] else "FAIL"
    print(f"{'CER (Total)':<25} {summary.total_cer:.1%}{'':<10} < {TARGETS['cer_clean']:.0%}{'':<10} {cer_status}")

    # Sample-level metrics
    print(f"\n{'WER (Average)':<25} {summary.avg_wer:.1%}")
    print(f"{'CER (Average)':<25} {summary.avg_cer:.1%}")

    if summary.samples_meeting_targets:
        print(f"\nSamples meeting targets:")
        print(f"  WER < 15%: {summary.samples_meeting_targets.get('wer_clean', 0)}/{summary.num_samples}")
        print(f"  CER < 10%: {summary.samples_meeting_targets.get('cer_clean', 0)}/{summary.num_samples}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate STT model")
    parser.add_argument("--model-path", required=True, help="Path to STT model")
    parser.add_argument("--test-set", required=True, help="Path to test manifest JSONL")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--output", help="Output path for results JSON")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load model
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        processor = WhisperProcessor.from_pretrained(args.model_path)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path)

        device = args.device if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded from {args.model_path}")
        logger.info(f"Using device: {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Evaluate
    results, summary = evaluate_test_set(
        model, processor, Path(args.test_set), device, args.max_samples
    )

    # Print report
    print_evaluation_report(summary)

    # Save results
    if args.output:
        output_data = {
            "summary": {
                "num_samples": summary.num_samples,
                "total_wer": summary.total_wer,
                "total_cer": summary.total_cer,
                "avg_wer": summary.avg_wer,
                "avg_cer": summary.avg_cer,
                "avg_rtf": summary.avg_rtf,
            },
            "results": [
                {
                    "audio_path": r.audio_path,
                    "reference": r.reference,
                    "prediction": r.prediction,
                    "wer": r.wer,
                    "cer": r.cer,
                }
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
