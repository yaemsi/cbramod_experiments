from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .train import FitResult
from .utils import save_json


CBRAMOD_PAPER_REFERENCE = {
    "balanced_accuracy": {"mean": 0.6370, "std": 0.0151},
    "auprc": {"mean": 0.7139, "std": 0.0088},
    "auroc": {"mean": 0.6988, "std": 0.0068},
}
# Backwards-compatible public name.
PAPER_REFERENCE = CBRAMOD_PAPER_REFERENCE


@dataclass(frozen=True)
class ReproductionRun:
    seed: int
    output_dir: str
    result: FitResult


def aggregate_reproduction_runs(
    runs: list[ReproductionRun],
    output_path: str | Path,
    *,
    model_name: str = "cbramod",
    paper_reference: dict[str, dict[str, float]] | None = CBRAMOD_PAPER_REFERENCE,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required")
    summary: dict[str, Any] = {
        "model": model_name,
        "num_runs": len(runs),
        "seeds": [run.seed for run in runs],
        "runs": [
            {
                "seed": run.seed,
                "output_dir": run.output_dir,
                **run.result.to_dict(),
            }
            for run in runs
        ],
    }
    if paper_reference is not None:
        summary["paper_reference"] = paper_reference

    aggregate: dict[str, dict[str, float]] = {}
    for name in ("balanced_accuracy", "auprc", "auroc", "average_precision"):
        values = np.asarray(
            [getattr(run.result.test, name) for run in runs], dtype=np.float64
        )
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if paper_reference is not None and name in paper_reference:
            aggregate[name]["delta_from_paper_mean"] = float(
                values.mean() - paper_reference[name]["mean"]
            )
    summary["test_aggregate"] = aggregate
    save_json(summary, output_path)
    return summary
