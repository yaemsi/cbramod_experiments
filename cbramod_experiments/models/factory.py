from __future__ import annotations

from torch import nn

from cbramod_experiments.utils.config import ModelConfig

from .cbramod import CBraModClassifier
from .eegsimpleconv import EEGSimpleConv


def build_model(config: ModelConfig) -> nn.Module:
    name = config.name.lower()
    if name == "cbramod":
        num_patches = int(config.sampling_rate * config.window_seconds) // 200
        return CBraModClassifier(
            num_channels=config.num_channels,
            num_patches=num_patches,
            classifier=config.classifier,
            pretrained=config.pretrained,
            checkpoint_repo=config.checkpoint_repo,
            checkpoint_filename=config.checkpoint_filename,
            freeze_backbone=config.freeze_backbone,
            **config.kwargs,
        )
    if name in {"eegsimpleconv", "simpleconv"}:
        return EEGSimpleConv(
            num_channels=config.num_channels,
            sampling_rate=config.sampling_rate,
            **config.kwargs,
        )
    raise ValueError(f"Unknown model name: {config.name}")
