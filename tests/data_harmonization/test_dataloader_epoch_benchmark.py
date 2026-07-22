from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cbramod_experiments.data_harmonization import ArrowShardWriter, EEGWindow
from cbramod_experiments.utils.data_benchmark import benchmark_dataloader_epoch


def _window(index: int, *, split: str = "train") -> EEGWindow:
    return EEGWindow(
        signal=np.full((2, 8), float(index), dtype=np.float32),
        sampling_rate_hz=4.0,
        channel_names=("C3", "C4"),
        channel_mask=np.ones(2, dtype=np.bool_),
        dataset_id="synthetic",
        subject_id=f"sub-{index:03d}",
        session_id="ses-01",
        task="benchmark",
        start_seconds=0.0,
        duration_seconds=2.0,
        label=index % 2,
        split=split,
        source_uri=f"synthetic-{index}",
        source_format="synthetic",
        sample_id=f"sample-{index:03d}",
        amplitude_scale=1.0,
    )


def _manifest(tmp_path: Path) -> Path:
    writer = ArrowShardWriter(
        tmp_path / "arrow",
        records_per_batch=3,
        batches_per_shard=2,
        overwrite=True,
    )
    for index in range(10):
        writer.add(_window(index))
    writer.close()
    return tmp_path / "arrow" / "manifest.parquet"


def test_full_epoch_benchmark_reads_every_streaming_example(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "epoch.json"
    result = benchmark_dataloader_epoch(
        manifest,
        backend="arrow_streaming",
        split="train",
        output_path=output,
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        streaming_shuffle_buffer_size=4,
        show_progress=False,
    )

    assert result.expected_examples == 10
    assert result.observed_examples == 10
    assert result.batches == 3
    assert result.complete_epoch
    assert result.first_batch_shape == (4, 2, 8)
    assert result.examples_per_second > 0
    assert result.epoch_seconds >= result.first_batch_seconds > 0
    assert json.loads(output.read_text())["complete_epoch"] is True


def test_full_epoch_benchmark_supports_random_access_arrow(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = benchmark_dataloader_epoch(
        manifest,
        backend="arrow",
        split="train",
        batch_size=6,
        num_workers=0,
        show_progress=False,
    )
    assert result.observed_examples == 10
    assert result.batches == 2
    assert result.complete_epoch
