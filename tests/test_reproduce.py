from __future__ import annotations

import json
from pathlib import Path

from cbramod_experiments.utils import (
    BinaryMetrics,
    FitResult,
    ReproductionRun,
    aggregate_reproduction_runs,
)


def _metrics(value: float) -> BinaryMetrics:
    return BinaryMetrics(
        balanced_accuracy=value,
        auprc=value + 0.05,
        auroc=value + 0.10,
        average_precision=value + 0.04,
        confusion_matrix=[[1, 0], [0, 1]],
        num_examples=2,
    )


def test_aggregate_reproduction_runs(tmp_path: Path) -> None:
    runs = [
        ReproductionRun(
            seed=seed,
            output_dir=f"seed_{seed}",
            result=FitResult(
                best_epoch=1,
                validation=_metrics(value),
                test=_metrics(value),
                checkpoint_path=f"seed_{seed}/best_model.pt",
            ),
        )
        for seed, value in ((1, 0.60), (2, 0.64))
    ]
    path = tmp_path / "summary.json"
    summary = aggregate_reproduction_runs(runs, path, model_name="cbramod")
    assert summary["model"] == "cbramod"
    assert summary["num_runs"] == 2
    assert summary["test_aggregate"]["balanced_accuracy"]["mean"] == 0.62
    assert summary["test_aggregate"]["balanced_accuracy"]["std"] > 0
    assert json.loads(path.read_text())["seeds"] == [1, 2]
