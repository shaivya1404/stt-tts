#!/usr/bin/env python3
"""Whisper fine-tuning with LoRA support for Indic languages STT.

Supports:
- Full fine-tuning or parameter-efficient LoRA fine-tuning
- Mixed precision training (fp16/bf16)
- Gradient checkpointing for memory efficiency
- Multiple Indic language support

Usage:
    python train_whisper.py --config configs/whisper_finetune_indic_v1.yaml
    python train_whisper.py --config configs/whisper_finetune_indic_v1.yaml --device cuda

Dependencies:
    pip install datasets transformers peft accelerate
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

try:
    from datasets import Audio, Dataset, load_dataset, load_from_disk
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
    )
    from transformers.trainer_utils import get_last_checkpoint
except ImportError as exc:
    raise ImportError(
        "Whisper training requires `datasets` and `transformers`. "
        "Install them via `pip install datasets transformers`."
    ) from exc

# Optional PEFT for LoRA
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not available. LoRA fine-tuning disabled. Install with: pip install peft")

# Optional accelerate
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

import evaluate
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper speech-to-text training."""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder_start_token_id if present at beginning
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_dataset(
    manifest_path: Path,
    processor: WhisperProcessor,
    sample_rate: int = 16000,
    audio_column: str = "audio_filepath",
    text_column: str = "text",
    max_input_length: float = 30.0,
    language: Optional[str] = None,
) -> Dataset:
    """Prepare dataset from manifest file.

    Args:
        manifest_path: Path to JSONL manifest
        processor: Whisper processor
        sample_rate: Target sample rate
        audio_column: Column name for audio paths
        text_column: Column name for transcripts
        max_input_length: Maximum audio length in seconds
        language: Language for forced decoding

    Returns:
        Processed HuggingFace Dataset
    """
    # Load dataset
    dataset = load_dataset("json", data_files={"data": str(manifest_path)})["data"]

    # Cast audio column
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))

    def prepare_example(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch[audio_column]

        # Skip if audio is too long
        if audio["array"].shape[0] / audio["sampling_rate"] > max_input_length:
            return {"input_features": None, "labels": None}

        # Compute input features
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]

        # Encode target text
        labels = processor.tokenizer(batch[text_column]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    # Process dataset
    dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    # Filter out None values (audio too long)
    dataset = dataset.filter(lambda x: x["input_features"] is not None)

    return dataset


def setup_lora(
    model: WhisperForConditionalGeneration,
    lora_config: Dict[str, Any],
) -> WhisperForConditionalGeneration:
    """Configure LoRA for the model.

    Args:
        model: Base Whisper model
        lora_config: LoRA configuration dict

    Returns:
        Model with LoRA adapters
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library required for LoRA. Install with: pip install peft")

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.get("lora_r", 32),
        lora_alpha=lora_config.get("lora_alpha", 64),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj",  # Attention projections
            "k_proj", "out_proj",  # Additional attention layers
            "fc1", "fc2",  # Feed-forward layers
        ]),
        inference_mode=False,
    )

    # Prepare model for k-bit training if needed
    if lora_config.get("use_8bit", False) or lora_config.get("use_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def compute_metrics(pred, processor: WhisperProcessor, metric):
    """Compute WER metric for evaluation."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on Indic manifests")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Optional torch device override (cpu/cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))
    print(f"Loaded config: {args.config}")

    # Training parameters
    sample_rate = int(config.get("sample_rate", 16000))
    audio_column = config.get("audio_column", "audio_filepath")
    text_column = config.get("text_column", "text")
    base_model = config["base_model"]
    training_config = config.get("training", {})

    # LoRA configuration
    use_lora = training_config.get("use_lora", False)
    lora_config = training_config.get("lora", {})

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor and model
    print(f"Loading model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if training_config.get("fp16", True) else torch.float32,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_config.get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # Apply LoRA if configured
    if use_lora:
        print("Applying LoRA adapters...")
        lora_params = {
            "lora_r": lora_config.get("r", training_config.get("lora_r", 32)),
            "lora_alpha": lora_config.get("alpha", training_config.get("lora_alpha", 64)),
            "lora_dropout": lora_config.get("dropout", training_config.get("lora_dropout", 0.1)),
            "target_modules": lora_config.get("target_modules", [
                "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"
            ]),
            "use_8bit": lora_config.get("use_8bit", False),
            "use_4bit": lora_config.get("use_4bit", False),
        }
        model = setup_lora(model, lora_params)

    # Load datasets
    print("Loading datasets...")
    train_dataset = prepare_dataset(
        Path(config["train_manifest"]),
        processor,
        sample_rate,
        audio_column,
        text_column,
    )
    print(f"Train dataset: {len(train_dataset)} examples")

    eval_dataset = prepare_dataset(
        Path(config["val_manifest"]),
        processor,
        sample_rate,
        audio_column,
        text_column,
    )
    print(f"Eval dataset: {len(eval_dataset)} examples")

    # Setup output directory
    output_dir = Path(settings.model_base_path) / "stt" / "whisper" / config["model_name"] / config["version"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Setup metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor, wer_metric)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        per_device_eval_batch_size=training_config.get("batch_size", 4),
        learning_rate=float(training_config.get("learning_rate", 1e-5)),
        num_train_epochs=int(training_config.get("epochs", 3)),
        max_steps=int(training_config.get("max_steps", -1)),
        warmup_steps=int(training_config.get("warmup_steps", 500)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
        logging_steps=int(training_config.get("log_every_n_steps", 50)),
        eval_strategy="steps",
        eval_steps=int(training_config.get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=int(config.get("output", {}).get("save_every_n_steps", 1000)),
        save_total_limit=3,
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=config.get("output", {}).get("push_to_hub", False),
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # Check for checkpoint to resume
    last_checkpoint = None
    if args.resume and (output_dir / "checkpoints").exists():
        last_checkpoint = get_last_checkpoint(str(output_dir / "checkpoints"))
        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")

    # Setup trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )

    # Training or evaluation
    if args.eval_only:
        print("Running evaluation...")
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
    else:
        print("Starting training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model
        print("Saving model...")
        if use_lora:
            # Save LoRA adapters separately
            model.save_pretrained(output_dir / "lora_adapters")
            # Also save merged model for inference
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir / "merged")
        else:
            model.save_pretrained(output_dir)

        processor.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

        # Final evaluation
        print("Running final evaluation...")
        metrics = trainer.evaluate()
        print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
