#!/usr/bin/env python3
"""Training skeleton for a Conformer RNNT-style STT model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings  # noqa: E402  # pylint: disable=wrong-import-position


class STTDataset(Dataset):
    """Tiny placeholder dataset that reads JSONL manifests when available."""

    def __init__(self, manifest_path: Path, sample_rate: int, sequence_length: int, feature_dim: int) -> None:
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.samples = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        if not self.manifest_path.exists():
            return [{"text": "hello world", "duration": 3.5}]
        data: List[Dict[str, Any]] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            try:
                data.append(json.loads(entry))
            except json.JSONDecodeError:
                data.append({"text": entry, "duration": 3.0})
        return data or [{"text": "fallback sample", "duration": 2.8}]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _sample = self.samples[index]
        sequence = torch.randn(self.sequence_length, self.feature_dim)
        labels = torch.randint(0, 128, (self.sequence_length,), dtype=torch.long)
        return sequence, labels


class ConformerRNNT(nn.Module):
    """Simplified encoder/classifier network used as RNNT stand-in."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, vocab_size: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # (batch, seq, feat)
        batch, seq_len, feat = inputs.shape
        x = inputs.view(batch * seq_len, feat)
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits.view(batch, seq_len, -1)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    training_cfg = config.get("training", {})
    sequence_length = int(training_cfg.get("sequence_length", 128))
    feature_dim = int(training_cfg.get("feature_dim", 80))
    sample_rate = int(config.get("sample_rate", 16000))

    train_dataset = STTDataset(
        Path(config["train_manifest"]),
        sample_rate,
        sequence_length,
        feature_dim,
    )
    val_dataset = STTDataset(
        Path(config["val_manifest"]),
        sample_rate,
        sequence_length,
        feature_dim,
    )

    train_loader = DataLoader(train_dataset, batch_size=training_cfg.get("batch_size", 4), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg.get("batch_size", 4))
    return train_loader, val_loader


def save_checkpoint(model: nn.Module, output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{label}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    (output_dir / "TRAINING_COMPLETE.txt").write_text("This directory stores placeholder checkpoints\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conformer RNNT placeholder")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Optional torch device override (cpu/cuda)")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(settings.model_base_path) / "stt" / config["model_name"] / config["version"]

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    train_loader, val_loader = prepare_dataloaders(config)
    training_cfg = config.get("training", {})
    feature_dim = int(training_cfg.get("feature_dim", 80))
    model = ConformerRNNT(input_dim=feature_dim).to(device)

    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 3e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(training_cfg.get("epochs", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every_n_steps", 10))

    for epoch in range(1, epochs + 1):
        model.train()
        for step, (features, targets) in enumerate(train_loader, start=1):
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        save_checkpoint(model, output_dir, f"checkpoint_epoch_{epoch}")

        # Tiny validation pass (placeholder)
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            batches = 0
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                logits = model(features)
                val_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += val_loss.item()
                batches += 1
            if batches:
                print(f"epoch={epoch} val_loss={total_loss / batches:.4f}")

    save_checkpoint(model, output_dir, "model_final")
    (output_dir / "config.used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Training artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
