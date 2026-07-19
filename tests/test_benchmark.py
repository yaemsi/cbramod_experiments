from pathlib import Path

import torch
from torch import nn

from cbramod_experiments.utils import benchmark_model


def test_benchmark_model_writes_json(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 1))
    output = tmp_path / "benchmark.json"
    result = benchmark_model(
        model,
        model_name="tiny",
        device=torch.device("cpu"),
        num_channels=2,
        num_points=4,
        batch_sizes=[1, 2],
        warmup_iterations=0,
        measured_iterations=2,
        output_path=output,
    )
    assert result.parameters == 9
    assert len(result.batches) == 2
    assert output.exists()
