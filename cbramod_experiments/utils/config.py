from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    path: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_channels: int = 32
    num_classes: int = 2
    sampling_rate: int = 200
    window_seconds: float = 4.0
    pretrained: bool = False
    checkpoint_repo: str = "weighting666/CBraMod"
    checkpoint_filename: str = "pretrained_weights.pth"
    classifier: str = "avg_pool"
    freeze_backbone: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    head_lr: float | None = 1e-3
    weight_decay: float = 5e-2
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    patience: int = 10
    seed: int = 3407
    amp: bool = True
    output_dir: str = "outputs/run"
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return ExperimentConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
