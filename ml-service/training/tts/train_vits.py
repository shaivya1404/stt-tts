"""VITS multi-speaker Indic training skeleton."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings  # noqa: E402  # pylint: disable=wrong-import-position

TEXT_EMBED_DIM = 256
SPEAKER_EMBED_DIM = 64
MEL_DIM = 80


class TTSDataset(Dataset[Any]):
    """Lightweight dataset that reads JSONL manifests or falls back to synthetic samples."""

    def __init__(self, manifest_path: Path | None, fallback_size: int = 128) -> None:
        self.manifest_path = manifest_path
        self.samples = self._load_manifest() if manifest_path and manifest_path.exists() else self._generate_fallback(fallback_size)

    def _load_manifest(self) -> List[Dict[str, Any]]:
        assert self.manifest_path is not None
        entries: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                data = json.loads(payload)
                entries.append(
                    {
                        "text": data.get("text", "hello world"),
                        "speaker_id": int(data.get("speaker_id", 0)),
                    }
                )
        if not entries:
            return self._generate_fallback(64)
        return entries

    def _generate_fallback(self, size: int) -> List[Dict[str, Any]]:
        vocab = ["hello", "namaste", "vanakkam", "welcome", "demo"]
        return [
            {"text": f"{random.choice(vocab)} {idx}", "speaker_id": idx % 10}  # noqa: S311 - deterministic-ish demo
            for idx in range(size)
        ]

    def __len__(self) -> int:  # noqa: D401 - Dataset protocol
        return len(self.samples)

    @staticmethod
    def _encode_text(text: str) -> torch.Tensor:
        vector = torch.zeros(TEXT_EMBED_DIM)
        for idx, byte in enumerate(text.encode("utf-8")):
            vector[idx % TEXT_EMBED_DIM] += byte / 255.0
        return vector

    @staticmethod
    def _speaker_vector(speaker_id: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(speaker_id)
        return torch.rand(SPEAKER_EMBED_DIM, generator=generator)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        text_embedding = self._encode_text(sample["text"])
        speaker_embedding = self._speaker_vector(sample["speaker_id"])
        mel_target = torch.rand(MEL_DIM)
        return {
            "text_embedding": text_embedding,
            "speaker_embedding": speaker_embedding,
            "mel_target": mel_target,
        }


class DummyVITSModel(nn.Module):
    """Tiny feed-forward network standing in for the full VITS stack."""

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.speaker_encoder = nn.Sequential(nn.Linear(SPEAKER_EMBED_DIM, 128), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, MEL_DIM),
        )

    def forward(self, text_embedding: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:  # noqa: D401
        text_latent = self.text_encoder(text_embedding)
        speaker_latent = self.speaker_encoder(speaker_embedding)
        combined = torch.cat([text_latent, speaker_latent], dim=-1)
        return self.decoder(combined)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(manifest_path: str | None, batch_size: int) -> DataLoader[Dict[str, torch.Tensor]]:
    dataset = TTSDataset(Path(manifest_path) if manifest_path else None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prepare_output_dirs(config: Dict[str, Any]) -> tuple[Path, Path]:
    output_cfg = config.get("output", {})
    base = Path(output_cfg.get("base_path") or (Path(settings.model_base_path) / "tts"))
    run_dir = base / config["model_name"] / config["version"]
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoints_dir


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int) -> None:
    payload = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the Indic VITS multi-speaker model (placeholder loop)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Optional torch device override (cpu/cuda)")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    epochs = int(training_cfg.get("epochs", 3))
    steps_per_epoch = int(training_cfg.get("steps_per_epoch", 50))
    log_every = int(training_cfg.get("log_every_n_steps", 5))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 50))

    learning_rate = float(config.get("optimizer", {}).get("learning_rate", 2e-4))
    weight_decay = float(config.get("optimizer", {}).get("weight_decay", 1e-2))

    device = torch.device(args.device or settings.device)
    model = DummyVITSModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = build_dataloader(config.get("train_manifest"), batch_size)
    _, checkpoints_dir = prepare_output_dirs(config)

    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            text_embedding = batch["text_embedding"].to(device)
            speaker_embedding = batch["speaker_embedding"].to(device)
            mel_target = batch["mel_target"].to(device)

            optimizer.zero_grad()
            prediction = model(text_embedding, speaker_embedding)
            loss = criterion(prediction, mel_target)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")

            if global_step % checkpoint_interval == 0:
                ckpt_path = checkpoints_dir / f"epoch{epoch}_step{global_step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step)

            if step >= steps_per_epoch:
                break

    final_path = checkpoints_dir.parent / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    train()
