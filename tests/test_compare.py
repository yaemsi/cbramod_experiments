import json
from pathlib import Path

from cbramod_experiments.utils import compare_experiments


def _summary(model: str, offset: float) -> dict[str, object]:
    return {
        "model": model,
        "num_runs": 5,
        "test_aggregate": {
            metric: {"mean": base + offset, "std": 0.01}
            for metric, base in {
                "balanced_accuracy": 0.60,
                "auprc": 0.65,
                "auroc": 0.67,
            }.items()
        },
    }


def test_compare_experiments_writes_reports(tmp_path: Path) -> None:
    cb = tmp_path / "cb.json"
    sc = tmp_path / "sc.json"
    cb.write_text(json.dumps(_summary("cbramod", 0.0)))
    sc.write_text(json.dumps(_summary("eegsimpleconv", 0.02)))

    result = compare_experiments(cb, sc, tmp_path / "report")
    assert result["metrics"]["auroc"]["winner"] == "eegsimpleconv"
    assert (tmp_path / "report" / "comparison.json").exists()
    assert (tmp_path / "report" / "comparison.md").exists()
