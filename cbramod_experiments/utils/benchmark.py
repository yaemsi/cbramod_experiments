from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .utils import count_parameters, save_json


@dataclass(frozen=True)
class BatchBenchmark:
    batch_size: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    throughput_examples_per_second: float
    peak_memory_mb: float | None


@dataclass(frozen=True)
class ModelBenchmark:
    model: str
    device: str
    parameters: int
    trainable_parameters: int
    state_size_mb: float
    input_shape: list[int]
    warmup_iterations: int
    measured_iterations: int
    batches: list[BatchBenchmark]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batches"] = [asdict(batch) for batch in self.batches]
        return payload


def benchmark_model(
    model: torch.nn.Module,
    *,
    model_name: str,
    device: torch.device,
    num_channels: int,
    num_points: int,
    batch_sizes: list[int],
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
    output_path: str | Path | None = None,
) -> ModelBenchmark:
    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and measured_iterations positive")
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")

    model = model.to(device).eval()
    batch_results: list[BatchBenchmark] = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, num_channels, num_points, device=device)
            for _ in range(warmup_iterations):
                model(inputs)
            _synchronize(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elapsed: list[float] = []
            for _ in range(measured_iterations):
                _synchronize(device)
                start = perf_counter()
                model(inputs)
                _synchronize(device)
                elapsed.append(perf_counter() - start)

            timings = np.asarray(elapsed, dtype=np.float64)
            mean_seconds = float(timings.mean())
            peak_memory = (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            )
            batch_results.append(
                BatchBenchmark(
                    batch_size=batch_size,
                    mean_latency_ms=mean_seconds * 1000,
                    median_latency_ms=float(np.median(timings) * 1000),
                    p95_latency_ms=float(np.percentile(timings, 95) * 1000),
                    throughput_examples_per_second=float(batch_size / mean_seconds),
                    peak_memory_mb=peak_memory,
                )
            )

    result = ModelBenchmark(
        model=model_name,
        device=str(device),
        parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, trainable_only=True),
        state_size_mb=_state_size_mb(model),
        input_shape=[num_channels, num_points],
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        batches=batch_results,
    )
    if output_path is not None:
        save_json(result.to_dict(), output_path)
    return result


def _state_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    return float(total_bytes / (1024**2))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
