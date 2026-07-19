from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import save_json


_METRICS = ("balanced_accuracy", "auprc", "auroc")
_DISPLAY_NAMES = {
    "balanced_accuracy": "Balanced accuracy",
    "auprc": "AUC-PR",
    "auroc": "AUROC",
}


def compare_experiments(
    cbramod_summary_path: str | Path,
    simpleconv_summary_path: str | Path,
    output_dir: str | Path,
    *,
    cbramod_benchmark_path: str | Path | None = None,
    simpleconv_benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    cbramod = _read_json(cbramod_summary_path)
    simpleconv = _read_json(simpleconv_summary_path)
    comparison: dict[str, Any] = {
        "cbramod_summary": str(cbramod_summary_path),
        "simpleconv_summary": str(simpleconv_summary_path),
        "num_runs": {
            "cbramod": int(cbramod["num_runs"]),
            "eegsimpleconv": int(simpleconv["num_runs"]),
        },
        "metrics": {},
    }

    for metric in _METRICS:
        cb = cbramod["test_aggregate"][metric]
        sc = simpleconv["test_aggregate"][metric]
        comparison["metrics"][metric] = {
            "cbramod_mean": float(cb["mean"]),
            "cbramod_std": float(cb["std"]),
            "eegsimpleconv_mean": float(sc["mean"]),
            "eegsimpleconv_std": float(sc["std"]),
            "simpleconv_minus_cbramod": float(sc["mean"] - cb["mean"]),
            "winner": "eegsimpleconv" if sc["mean"] > cb["mean"] else "cbramod",
        }

    benchmarks: dict[str, Any] = {}
    if cbramod_benchmark_path is not None:
        benchmarks["cbramod"] = _read_json(cbramod_benchmark_path)
    if simpleconv_benchmark_path is not None:
        benchmarks["eegsimpleconv"] = _read_json(simpleconv_benchmark_path)
    if benchmarks:
        comparison["benchmarks"] = benchmarks

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(comparison, output_dir / "comparison.json")
    (output_dir / "comparison.md").write_text(
        _render_markdown(comparison), encoding="utf-8"
    )
    return comparison


def _render_markdown(comparison: dict[str, Any]) -> str:
    rows: list[str] = []
    for metric in _METRICS:
        values = comparison["metrics"][metric]
        rows.append(
            f"| {_DISPLAY_NAMES[metric]} | "
            f"{values['cbramod_mean']:.4f} ± {values['cbramod_std']:.4f} | "
            f"{values['eegsimpleconv_mean']:.4f} ± {values['eegsimpleconv_std']:.4f} | "
            f"{values['simpleconv_minus_cbramod']:+.4f} |"
        )

    benchmark_section = ""
    benchmarks = comparison.get("benchmarks", {})
    if {"cbramod", "eegsimpleconv"}.issubset(benchmarks):
        cb = benchmarks["cbramod"]
        sc = benchmarks["eegsimpleconv"]
        benchmark_rows = [
            f"| Parameters | {cb['parameters']:,} | {sc['parameters']:,} |",
            f"| State size (MiB) | {cb['state_size_mb']:.2f} | {sc['state_size_mb']:.2f} |",
        ]
        cb_batches = {item["batch_size"]: item for item in cb["batches"]}
        sc_batches = {item["batch_size"]: item for item in sc["batches"]}
        for batch_size in sorted(set(cb_batches).intersection(sc_batches)):
            benchmark_rows.append(
                f"| Batch {batch_size} mean latency (ms) | "
                f"{cb_batches[batch_size]['mean_latency_ms']:.3f} | "
                f"{sc_batches[batch_size]['mean_latency_ms']:.3f} |"
            )
            benchmark_rows.append(
                f"| Batch {batch_size} throughput (examples/s) | "
                f"{cb_batches[batch_size]['throughput_examples_per_second']:.1f} | "
                f"{sc_batches[batch_size]['throughput_examples_per_second']:.1f} |"
            )
        benchmark_section = (
            "\n## Efficiency\n\n"
            "| Measurement | CBraMod | EEGSimpleConv |\n"
            "|---|---:|---:|\n"
            + "\n".join(benchmark_rows)
            + "\n"
        )

    return (
        "# Task C: CBraMod versus EEGSimpleConv on SHU-MI\n\n"
        "Both models use the same processed examples, subject split, binary loss, "
        "validation-AUROC checkpoint selection, and test metric implementation. "
        "EEGSimpleConv uses its architecture-specific Adam schedule; model-specific "
        "normalization tricks, Euclidean alignment, mixup, and subject-identity "
        "regularization are intentionally excluded to keep this an architecture-focused "
        "comparison.\n\n"
        "## Predictive performance\n\n"
        "| Metric | CBraMod | EEGSimpleConv | SimpleConv − CBraMod |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n"
        + benchmark_section
        + "\n## Interpretation checklist\n\n"
        "Discuss absolute performance, run-to-run variance, early overfitting, parameter "
        "count, latency, throughput, memory, deployment constraints, and the value of "
        "pretraining. Prefer CBraMod when transfer from large-scale pretraining improves "
        "accuracy or label efficiency; prefer EEGSimpleConv when a compact, transparent, "
        "low-latency model gives comparable performance.\n"
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload
