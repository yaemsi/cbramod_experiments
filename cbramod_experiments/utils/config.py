from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class DataConfig:
    path: str
    backend: Literal["hdf5", "arrow"] = "hdf5"
    batch_size: int = 64
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
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = (
        "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
    )
    classifier: str = "avg_pool"
    freeze_backbone: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    head_lr: float | None = 5e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0
    patience: int = 50
    seed: int = 3407
    amp: bool = True
    output_dir: str = "outputs/run"
    device: str = "auto"
    optimizer: Literal["adam", "adamw"] = "adamw"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    scheduler_interval: Literal["step", "epoch"] = "step"
    min_lr: float = 1e-6
    lr_decay_epoch: int = 40
    lr_decay_gamma: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = ExperimentConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.num_classes != 2:
        raise ValueError("The SHU-MI experiments currently support exactly two classes")
    if config.data.backend not in {"hdf5", "arrow"}:
        raise ValueError("data.backend must be 'hdf5' or 'arrow'")
    if config.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.training.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.training.patience <= 0:
        raise ValueError("patience must be positive")
    if config.training.lr <= 0:
        raise ValueError("lr must be positive")
    if config.training.head_lr is not None and config.training.head_lr <= 0:
        raise ValueError("head_lr must be positive when provided")
    if not 0.0 <= config.training.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if config.training.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if config.training.optimizer.lower() not in {"adam", "adamw"}:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    if config.training.scheduler.lower() not in {"cosine", "step", "none"}:
        raise ValueError("scheduler must be 'cosine', 'step', or 'none'")
    if config.training.scheduler_interval.lower() not in {"step", "epoch"}:
        raise ValueError("scheduler_interval must be 'step' or 'epoch'")
    if config.training.min_lr < 0:
        raise ValueError("min_lr cannot be negative")
    if config.training.lr_decay_epoch <= 0:
        raise ValueError("lr_decay_epoch must be positive")
    if not 0 < config.training.lr_decay_gamma <= 1:
        raise ValueError("lr_decay_gamma must be in (0, 1]")

    points = config.model.sampling_rate * config.model.window_seconds
    if not float(points).is_integer():
        raise ValueError("sampling_rate * window_seconds must be an integer")
    if config.model.name.lower() == "cbramod" and int(points) % 200:
        raise ValueError(
            "CBraMod requires a window containing a multiple of 200 samples"
        )
