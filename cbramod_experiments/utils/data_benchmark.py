from __future__ import annotations

import json
import time
from collections.abc import Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data_harmonization.datamodule import DataBackend, EEGDataModule
from ..data_harmonization.storage import StreamingArrowEEGDataset


@runtime_checkable
class SupportsSetEpoch(Protocol):
    """Structural type for datasets and samplers with epoch-aware shuffling."""

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch used to derive deterministic shuffle state."""


def _set_epoch_if_supported(obj: object | None, epoch: int) -> None:
    """Call ``set_epoch`` when an object implements the expected protocol."""

    if isinstance(obj, SupportsSetEpoch):
        obj.set_epoch(epoch)


def _expected_dataset_examples(dataset: object) -> int:
    """Return the number of examples assigned to the current process.

    Streaming datasets expose an explicit rank-aware count. Map-style datasets
    are accepted through the standard ``Sized`` protocol. Keeping the checks in
    this helper gives Pyright a sound narrowing before ``len`` is called.
    """

    if isinstance(dataset, StreamingArrowEEGDataset):
        return int(dataset.assigned_example_count())

    if isinstance(dataset, Sized):
        return len(dataset)

    raise TypeError(
        "Cannot determine the expected number of examples for dataset type "
        f"{type(dataset).__name__!r}"
    )


@dataclass(frozen=True)
class StreamingBenchmarkResult:
    manifest_path: str
    device: str
    batch_size: int
    num_workers: int
    prefetch_factor: int
    shuffle_buffer_size: int
    warmup_batches: int
    measured_batches: int
    examples: int
    signal_bytes: int
    elapsed_seconds: float
    examples_per_second: float
    mebibytes_per_second: float
    gibibits_per_second: float
    selected_examples: int
    selected_shards: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_streaming_dataset(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    warmup_batches: int = 10,
    max_batches: int = 200,
    shuffle_buffer_size: int = 2048,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dataset_ids: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    split: str | None = None,
    sampling_rate_hz: float | None = None,
    num_channels: int | None = None,
    num_samples: int | None = None,
    require_labels: bool = False,
) -> StreamingBenchmarkResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if warmup_batches < 0 or max_batches < 0:
        raise ValueError("warmup_batches and max_batches must be non-negative")

    resolved_device = torch.device(device)
    dataset = StreamingArrowEEGDataset(
        manifest_path,
        split=split,
        dataset_ids=dataset_ids,
        tasks=tasks,
        sampling_rate_hz=sampling_rate_hz,
        num_channels=num_channels,
        num_samples=num_samples,
        require_labels=require_labels,
        shuffle_shards=True,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    if num_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=resolved_device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=resolved_device.type == "cuda",
            persistent_workers=False,
            drop_last=False,
        )

    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            signals, labels = next(iterator)
        except StopIteration:
            break
        if resolved_device.type != "cpu":
            signals = signals.to(resolved_device, non_blocking=True)
            labels = labels.to(resolved_device, non_blocking=True)
    _synchronize(resolved_device)

    measured_batches = 0
    examples = 0
    signal_bytes = 0
    start = time.perf_counter()
    while max_batches == 0 or measured_batches < max_batches:
        try:
            signals, labels = next(iterator)
        except StopIteration:
            break
        if resolved_device.type != "cpu":
            signals = signals.to(resolved_device, non_blocking=True)
            labels = labels.to(resolved_device, non_blocking=True)
        measured_batches += 1
        examples += int(signals.shape[0])
        signal_bytes += signals.numel() * signals.element_size()
    _synchronize(resolved_device)
    elapsed = time.perf_counter() - start
    if measured_batches == 0 or elapsed <= 0:
        raise RuntimeError("No streaming batches were measured")

    mib_per_second = signal_bytes / elapsed / (1024**2)
    result = StreamingBenchmarkResult(
        manifest_path=str(manifest_path),
        device=str(resolved_device),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle_buffer_size=shuffle_buffer_size,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        examples=examples,
        signal_bytes=signal_bytes,
        elapsed_seconds=elapsed,
        examples_per_second=examples / elapsed,
        mebibytes_per_second=mib_per_second,
        gibibits_per_second=mib_per_second * 8 / 1024,
        selected_examples=len(dataset),
        selected_shards=dataset.num_shards,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class DataLoaderEpochBenchmarkResult:
    data_path: str
    backend: str
    split: str | None
    device: str
    epoch: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    streaming_shuffle_buffer_size: int
    expected_examples: int
    observed_examples: int
    batches: int
    signal_bytes: int
    loader_build_seconds: float
    iterator_startup_seconds: float
    first_batch_seconds: float
    epoch_seconds: float
    steady_state_seconds: float
    examples_per_second: float
    mebibytes_per_second: float
    first_batch_shape: tuple[int, ...]
    signal_dtype: str
    complete_epoch: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_dataloader_epoch(
    data_path: str | Path,
    *,
    backend: DataBackend = "arrow_streaming",
    split: str | None = "train",
    output_path: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    streaming_shuffle_buffer_size: int = 2048,
    seed: int = 0,
    epoch: int = 0,
    device: str | torch.device = "cpu",
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    show_progress: bool = True,
    require_labels: bool = True,
) -> DataLoaderEpochBenchmarkResult:
    """Measure one complete dataloader epoch, including worker startup.

    This benchmark intentionally excludes model forward/backward computation. It
    measures manifest filtering, worker startup, shard reads, decompression,
    collation, and optional host-to-device transfer. The reported payload
    throughput counts signal tensor bytes only, not filesystem metadata or
    compression overhead.
    """

    if split not in {"train", "val", "test", None}:
        raise ValueError(f"Unknown split: {split}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if streaming_shuffle_buffer_size <= 0:
        raise ValueError("streaming_shuffle_buffer_size must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    resolved_device = torch.device(device)
    resolved_pin_memory = (
        resolved_device.type == "cuda" if pin_memory is None else pin_memory
    )
    resolved_persistent_workers = persistent_workers and num_workers > 0

    build_start = time.perf_counter()
    module = EEGDataModule(
        data_path,
        backend=backend,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent_workers,
        seed=seed,
        streaming_shuffle_buffer_size=streaming_shuffle_buffer_size,
        prefetch_factor=prefetch_factor,
        require_labels=require_labels,
    )
    loader = module.loader(split)
    loader_build_seconds = time.perf_counter() - build_start

    dataset: object = loader.dataset
    sampler: object | None = getattr(loader, "sampler", None)
    _set_epoch_if_supported(dataset, epoch)
    _set_epoch_if_supported(sampler, epoch)

    expected_examples = _expected_dataset_examples(dataset)
    if expected_examples <= 0:
        raise RuntimeError("The selected dataloader view contains no examples")

    epoch_start = time.perf_counter()
    iterator = iter(loader)
    iterator_startup_seconds = time.perf_counter() - epoch_start

    observed_examples = 0
    batches = 0
    signal_bytes = 0
    first_batch_seconds = 0.0
    first_batch_shape: tuple[int, ...] = ()
    signal_dtype = ""

    rank_zero = (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )
    progress = tqdm(
        total=expected_examples,
        desc=f"{backend}:{split} epoch {epoch}",
        unit="example",
        disable=not show_progress or not rank_zero,
    )
    try:
        for signals, labels in iterator:
            if resolved_device.type != "cpu":
                signals = signals.to(resolved_device, non_blocking=True)
                labels = labels.to(resolved_device, non_blocking=True)

            batches += 1
            observed_examples += int(signals.shape[0])
            signal_bytes += signals.numel() * signals.element_size()
            if batches == 1:
                _synchronize(resolved_device)
                first_batch_seconds = time.perf_counter() - epoch_start
                first_batch_shape = tuple(int(value) for value in signals.shape)
                signal_dtype = str(signals.dtype)
            progress.update(int(signals.shape[0]))
    finally:
        progress.close()

    _synchronize(resolved_device)
    epoch_seconds = time.perf_counter() - epoch_start
    if batches == 0 or epoch_seconds <= 0:
        raise RuntimeError("No dataloader batches were measured")

    steady_state_seconds = max(0.0, epoch_seconds - first_batch_seconds)
    mib_per_second = signal_bytes / epoch_seconds / (1024**2)
    result = DataLoaderEpochBenchmarkResult(
        data_path=str(data_path),
        backend=backend,
        split=split,
        device=str(resolved_device),
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent_workers,
        streaming_shuffle_buffer_size=streaming_shuffle_buffer_size,
        expected_examples=expected_examples,
        observed_examples=observed_examples,
        batches=batches,
        signal_bytes=signal_bytes,
        loader_build_seconds=loader_build_seconds,
        iterator_startup_seconds=iterator_startup_seconds,
        first_batch_seconds=first_batch_seconds,
        epoch_seconds=epoch_seconds,
        steady_state_seconds=steady_state_seconds,
        examples_per_second=observed_examples / epoch_seconds,
        mebibytes_per_second=mib_per_second,
        first_batch_shape=first_batch_shape,
        signal_dtype=signal_dtype,
        complete_epoch=observed_examples == expected_examples,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result
