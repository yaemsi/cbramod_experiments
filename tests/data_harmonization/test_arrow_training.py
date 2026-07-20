from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cbramod_experiments.data_harmonization import ArrowShardWriter, EEGDataModule
from cbramod_experiments.data_harmonization.schema import EEGWindow
from cbramod_experiments.utils.train import fit_binary_classifier


def _window(index: int, split: str, label: int) -> EEGWindow:
    rng = np.random.default_rng(index)
    signal = rng.normal(loc=float(label), scale=0.2, size=(2, 20)).astype(np.float32)
    return EEGWindow(
        signal=signal,
        sampling_rate_hz=10.0,
        channel_names=("C3", "C4"),
        channel_mask=np.ones(2, dtype=np.bool_),
        dataset_id="shu-mi",
        subject_id=f"sub-{index + 1:03d}",
        session_id="ses-01",
        task="motorimagery",
        start_seconds=0.0,
        duration_seconds=2.0,
        label=label,
        split=split,
        source_uri="synthetic",
        source_format="synthetic",
        sample_id=f"sample-{split}-{index}",
        amplitude_scale=1.0,
    )


def test_training_loop_accepts_arrow_backend(tmp_path: Path) -> None:
    output = tmp_path / "arrow"
    writer = ArrowShardWriter(output, records_per_batch=4, overwrite=True)
    for index in range(12):
        writer.add(_window(index, "train", index % 2))
    for index in range(4):
        writer.add(_window(100 + index, "val", index % 2))
        writer.add(_window(200 + index, "test", index % 2))
    writer.close()

    loaders = EEGDataModule(
        output / "manifest.parquet",
        backend="arrow",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=1,
    ).loaders()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(40, 1))
    result = fit_binary_classifier(
        model,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        torch.device("cpu"),
        epochs=1,
        lr=1e-2,
        head_lr=None,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        patience=1,
        amp=False,
        output_dir=tmp_path / "run",
        scheduler_name="none",
    )
    assert result.best_epoch == 1
    assert Path(result.checkpoint_path).exists()
