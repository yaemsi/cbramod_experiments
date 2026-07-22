from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cbramod_experiments.data_harmonization import (
    ArrowShardWriter,
    EEGDataModule,
    EEGWindow,
    StreamingArrowEEGDataset,
)
from cbramod_experiments.utils.data_benchmark import benchmark_streaming_dataset


def _window(index: int, *, split: str = "train") -> EEGWindow:
    signal = np.full((2, 8), float(index), dtype=np.float32)
    return EEGWindow(
        signal=signal,
        sampling_rate_hz=4.0,
        channel_names=("C3", "C4"),
        channel_mask=np.ones(2, dtype=np.bool_),
        dataset_id="synthetic",
        subject_id=f"sub-{index:03d}",
        session_id="ses-01",
        task="streaming",
        start_seconds=0.0,
        duration_seconds=2.0,
        label=index % 2,
        split=split,
        source_uri=f"synthetic-{index}",
        source_format="synthetic",
        sample_id=f"sample-{index:03d}",
        amplitude_scale=1.0,
    )


def _dataset(tmp_path: Path, count: int = 24) -> Path:
    output = tmp_path / "arrow"
    writer = ArrowShardWriter(
        output,
        records_per_batch=2,
        batches_per_shard=2,
        overwrite=True,
    )
    for index in range(count):
        writer.add(_window(index))
    for offset, split in ((100, "val"), (200, "test")):
        for index in range(4):
            writer.add(_window(offset + index, split=split))
    writer.close()
    return output / "manifest.parquet"


def _ids(dataset: StreamingArrowEEGDataset) -> list[int]:
    return [int(signal[0, 0].item()) for signal, _ in dataset]


def test_streaming_dataset_visits_each_example_once_and_changes_epoch(
    tmp_path: Path,
) -> None:
    manifest = _dataset(tmp_path)
    dataset = StreamingArrowEEGDataset(
        manifest,
        split="train",
        require_labels=True,
        shuffle_shards=True,
        shuffle_buffer_size=5,
        seed=17,
    )

    dataset.set_epoch(0)
    first = _ids(dataset)
    dataset.set_epoch(1)
    second = _ids(dataset)

    assert sorted(first) == list(range(24))
    assert sorted(second) == list(range(24))
    assert first != second
    assert dataset.num_shards == 6


def test_streaming_dataset_partitions_shards_across_ranks(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    rank_zero = StreamingArrowEEGDataset(
        manifest,
        split="train",
        shuffle_shards=False,
        shuffle_buffer_size=0,
        rank=0,
        world_size=2,
    )
    rank_one = StreamingArrowEEGDataset(
        manifest,
        split="train",
        shuffle_shards=False,
        shuffle_buffer_size=0,
        rank=1,
        world_size=2,
    )

    zero_ids = set(_ids(rank_zero))
    one_ids = set(_ids(rank_one))
    assert zero_ids.isdisjoint(one_ids)
    assert zero_ids | one_ids == set(range(24))


def test_streaming_dataset_partitions_across_dataloader_workers(
    tmp_path: Path,
) -> None:
    manifest = _dataset(tmp_path)
    dataset = StreamingArrowEEGDataset(
        manifest,
        split="train",
        shuffle_shards=False,
        shuffle_buffer_size=0,
    )
    loader = DataLoader(dataset, batch_size=3, num_workers=2)
    observed: list[int] = []
    for signals, _ in loader:
        observed.extend(int(value) for value in signals[:, 0, 0].tolist())
    assert sorted(observed) == list(range(24))


def test_streaming_backend_works_through_data_module(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    loaders = EEGDataModule(
        manifest,
        backend="arrow_streaming",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=3,
        streaming_shuffle_buffer_size=5,
    ).loaders()
    signals, labels = next(iter(loaders["train"]))
    assert signals.shape == (4, 2, 8)
    assert labels.shape == (4,)


def test_streaming_throughput_benchmark_writes_json(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    output = tmp_path / "throughput.json"
    result = benchmark_streaming_dataset(
        manifest,
        output_path=output,
        batch_size=4,
        num_workers=0,
        warmup_batches=1,
        max_batches=3,
        shuffle_buffer_size=4,
        split="train",
        require_labels=True,
    )
    assert result.measured_batches == 3
    assert result.examples == 12
    assert result.signal_bytes == 12 * 2 * 8 * torch.tensor(0.0).element_size()
    assert result.examples_per_second > 0
    assert output.exists()
