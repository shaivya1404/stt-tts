#!/usr/bin/env python3
"""XTTS v2 fine-tuning for Indic languages.

Supports:
- Fine-tuning XTTS v2 on custom Indic language datasets
- Multi-speaker training
- Voice cloning optimization
- Memory-efficient training for Colab T4

Usage:
    python train_xtts.py --config configs/xtts_indic_v1.yaml
    python train_xtts.py --config configs/xtts_indic_v1.yaml --resume

Dependencies:
    pip install TTS torch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Coqui TTS not available. Install with: pip install TTS")

try:
    from torch.utils.data import DataLoader, Dataset
    import torchaudio
except ImportError as e:
    print(f"PyTorch/torchaudio not available: {e}")

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


class XTTSDataset(Dataset):
    """Dataset for XTTS fine-tuning."""

    def __init__(
        self,
        manifest_path: Path,
        max_audio_length: float = 11.0,
        min_audio_length: float = 1.0,
        sample_rate: int = 22050,
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to JSONL manifest
            max_audio_length: Maximum audio length in seconds
            min_audio_length: Minimum audio length in seconds
            sample_rate: Target sample rate
        """
        self.manifest_path = manifest_path
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.sample_rate = sample_rate
        self.samples = []

        self._load_manifest()

    def _load_manifest(self):
        """Load and filter samples from manifest."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)

                # Filter by duration
                duration = sample.get("duration", 0)
                if self.min_audio_length <= duration <= self.max_audio_length:
                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {self.manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load audio
        audio_path = sample["audio_filepath"]
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)

        except Exception as e:
            print(f"Failed to load audio {audio_path}: {e}")
            waveform = torch.zeros(self.sample_rate)  # Fallback

        return {
            "audio": waveform,
            "text": sample["text"],
            "language": sample.get("language", "hi-IN"),
            "speaker_id": sample.get("speaker_id", "default"),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for XTTS training."""
    # Find max audio length in batch
    max_len = max(item["audio"].shape[0] for item in batch)

    # Pad audio
    audio_batch = []
    audio_lengths = []
    for item in batch:
        audio = item["audio"]
        audio_lengths.append(audio.shape[0])
        if audio.shape[0] < max_len:
            padding = torch.zeros(max_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        audio_batch.append(audio)

    return {
        "audio": torch.stack(audio_batch),
        "audio_lengths": torch.tensor(audio_lengths),
        "text": [item["text"] for item in batch],
        "language": [item["language"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
    }


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model(config: Dict[str, Any], device: str) -> Xtts:
    """Setup XTTS model for fine-tuning.

    Args:
        config: Training configuration
        device: Torch device

    Returns:
        XTTS model
    """
    if not TTS_AVAILABLE:
        raise RuntimeError("Coqui TTS required. Install with: pip install TTS")

    base_model = config.get("base_model", "tts_models/multilingual/multi-dataset/xtts_v2")

    print(f"Loading base model: {base_model}")

    # Download and load model
    model_manager = ModelManager()

    # Check if it's a local path or model name
    if os.path.exists(base_model):
        model_path = base_model
        config_path = os.path.join(base_model, "config.json")
    else:
        # Download from TTS model hub
        model_path, config_path, _ = model_manager.download_model(base_model)

    # Load config
    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)

    # Initialize model
    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(xtts_config, checkpoint_dir=model_path)

    # Move to device
    model = model.to(device)

    # Enable training mode
    model.train()

    # Freeze layers if specified
    freeze_layers = config.get("training", {}).get("freeze_layers", [])
    for name, param in model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False
            print(f"Frozen: {name}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    return model


def train_epoch(
    model: Xtts,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: Dict[str, Any],
) -> float:
    """Train for one epoch.

    Args:
        model: XTTS model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Torch device
        epoch: Current epoch number
        config: Training config

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    gradient_accumulation = config.get("training", {}).get("gradient_accumulation", 1)
    use_amp = config.get("training", {}).get("mixed_precision", "fp16") == "fp16"

    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        audio = batch["audio"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        texts = batch["text"]
        languages = batch["language"]

        try:
            # Forward pass with mixed precision
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    # XTTS training forward
                    loss = model.forward_train(
                        text_tokens=texts,
                        audio=audio,
                        audio_lengths=audio_lengths,
                        languages=languages,
                    )
            else:
                loss = model.forward_train(
                    text_tokens=texts,
                    audio=audio,
                    audio_lengths=audio_lengths,
                    languages=languages,
                )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % gradient_accumulation == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            num_batches += 1

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / max(num_batches, 1)


def evaluate(
    model: Xtts,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Evaluate model.

    Args:
        model: XTTS model
        dataloader: Evaluation dataloader
        device: Torch device

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            texts = batch["text"]
            languages = batch["language"]

            try:
                loss = model.forward_train(
                    text_tokens=texts,
                    audio=audio,
                    audio_lengths=audio_lengths,
                    languages=languages,
                )
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune XTTS v2 for Indic languages")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))
    print(f"Loaded config: {args.config}")

    # Setup device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup model
    model = setup_model(config, device)

    # Setup datasets
    train_dataset = XTTSDataset(
        Path(config["train_manifest"]),
        max_audio_length=config.get("max_audio_length", 11.0),
        min_audio_length=config.get("min_audio_length", 1.0),
        sample_rate=config.get("sample_rate", 22050),
    )

    val_dataset = XTTSDataset(
        Path(config["val_manifest"]),
        max_audio_length=config.get("max_audio_length", 11.0),
        min_audio_length=config.get("min_audio_length", 1.0),
        sample_rate=config.get("sample_rate", 22050),
    )

    # Setup dataloaders
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Setup optimizer
    learning_rate = float(training_config.get("learning_rate", 1e-5))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Setup output directory
    output_dir = Path(config.get("output_dir", "outputs/xtts_indic"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    # Training loop
    epochs = training_config.get("epochs", 100)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")

        # Save latest
        latest_path = output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
