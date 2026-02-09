"""HiFi-GAN vocoder training skeleton for Indic TTS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from common import settings  # noqa: E402  # pylint: disable=wrong-import-position

MEL_DIM = 80
WAVEFORM_SAMPLES = 1024


class VocoderDataset(Dataset[Any]):
    """Generates mel/waveform pairs from a JSONL manifest or a synthetic fallback."""

    def __init__(self, manifest_path: Path | None, fallback_size: int = 256) -> None:
        self.manifest_path = manifest_path
        self.records = self._load_manifest() if manifest_path and manifest_path.exists() else self._generate_fallback(fallback_size)

    def _load_manifest(self) -> list[Dict[str, Any]]:
        assert self.manifest_path is not None
        items: list[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                data = json.loads(payload)
                items.append(
                    {
                        "mel_path": data.get("mel_path"),
                        "audio_path": data.get("audio_path"),
                    }
                )
        if not items:
            return self._generate_fallback(128)
        return items

    def _generate_fallback(self, size: int) -> list[Dict[str, Any]]:
        return [{"mel_path": None, "audio_path": None} for _ in range(size)]

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _record = self.records[index]
        mel = torch.rand(MEL_DIM)
        waveform = torch.rand(WAVEFORM_SAMPLES)
        return {"mel": mel, "waveform": waveform}


class DummyHiFiGAN(nn.Module):
    """Simple MLP representing the vocoder."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MEL_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, WAVEFORM_SAMPLES),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(mel)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_output_dir(config: Dict[str, Any]) -> Path:
    output_cfg = config.get("output", {})
    base = Path(output_cfg.get("base_path") or (Path(settings.model_base_path) / "tts"))
    vocoder_dir = base / config["model_name"] / config["version"] / "hifigan"
    vocoder_dir.mkdir(parents=True, exist_ok=True)
    return vocoder_dir


def train() -> None:
    parser = argparse.ArgumentParser(description="Train placeholder HiFi-GAN vocoder")
    parser.add_argument("--config", required=True, help="Path to YAML config (reuses the VITS config)")
    parser.add_argument("--device", default=None, help="Optional torch device override")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 3))
    steps_per_epoch = int(training_cfg.get("steps_per_epoch", 100))
    log_every = int(training_cfg.get("log_every_n_steps", 10))

    learning_rate = float(config.get("optimizer", {}).get("learning_rate", 2e-4))
    dataset = VocoderDataset(Path(config.get("train_manifest")) if config.get("train_manifest") else None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(args.device or settings.device)
    model = DummyHiFiGAN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    output_dir = prepare_output_dir(config)

    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(dataloader, start=1):
            mel = batch["mel"].to(device)
            waveform = batch["waveform"].to(device)

            optimizer.zero_grad()
            prediction = model(mel)
            loss = criterion(prediction, waveform)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % log_every == 0:
                print(f"[vocoder] epoch={epoch} step={global_step} loss={loss.item():.4f}")

            if step >= steps_per_epoch:
                break

    final_path = output_dir / "hifigan_placeholder.pt"
    torch.save(model.state_dict(), final_path)
    print(f"HiFi-GAN placeholder saved to {final_path}")


if __name__ == "__main__":
    train()
