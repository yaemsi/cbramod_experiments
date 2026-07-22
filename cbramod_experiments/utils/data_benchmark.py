from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from ..data_harmonization.storage import StreamingArrowEEGDataset


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
