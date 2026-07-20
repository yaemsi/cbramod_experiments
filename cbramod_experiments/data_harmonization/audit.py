from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..datasets.shumi import SHUDatasetAudit


def _subject_number(value: str) -> int:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse numeric subject ID from {value!r}")
    return int(digits)


def audit_arrow_shu(
    manifest_path: str | Path,
    *,
    require_complete_protocol: bool = False,
) -> SHUDatasetAudit:
    path = Path(manifest_path)
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    dataset_ids = {str(row["dataset_id"]) for row in rows}
    if dataset_ids != {"shu-mi"}:
        raise ValueError(
            f"SHU audit requires only dataset_id='shu-mi', got {sorted(dataset_ids)}"
        )

    expected_subjects = {
        "train": set(range(1, 16)),
        "val": set(range(16, 21)),
        "test": set(range(21, 26)),
    }
    warnings: list[str] = []
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest contains duplicate sample IDs")

    shapes = {(int(row["num_channels"]), int(row["num_samples"])) for row in rows}
    if len(shapes) != 1:
        raise ValueError(
            f"SHU training examples must share one shape, got {sorted(shapes)}"
        )
    channels, points = next(iter(shapes))
    split_examples: dict[str, int] = {}
    split_subjects: dict[str, list[int]] = {}
    split_class_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        subjects = sorted(
            {_subject_number(str(row["subject_id"])) for row in split_rows}
        )
        invalid = set(subjects).difference(expected_subjects[split])
        if invalid:
            raise ValueError(
                f"Subject leakage in {split}: subjects {sorted(invalid)} belong elsewhere"
            )
        labels = np.asarray([row["label"] for row in split_rows], dtype=np.int64)
        if labels.size and not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Split {split} contains non-binary labels")
        counts = np.bincount(labels, minlength=2) if labels.size else np.array([0, 0])
        split_examples[split] = len(split_rows)
        split_subjects[split] = subjects
        split_class_counts[split] = {"0": int(counts[0]), "1": int(counts[1])}
        missing = expected_subjects[split].difference(subjects)
        if missing:
            warnings.append(f"Split {split} is missing subjects {sorted(missing)}")
        if split_rows and (counts == 0).any():
            warnings.append(f"Split {split} does not contain both classes")

    unspecified = [row for row in rows if row["split"] not in {"train", "val", "test"}]
    if unspecified:
        raise ValueError(
            f"SHU manifest contains {len(unspecified)} examples without a valid split"
        )

    examples = len(rows)
    complete = all(
        set(split_subjects[name]) == expected_subjects[name]
        for name in expected_subjects
    )
    if examples != 11_988:
        warnings.append(f"Paper reports 11,988 examples; manifest contains {examples}")
    if channels != 32:
        warnings.append(
            f"Paper protocol uses 32 channels; manifest contains {channels}"
        )
    if points != 800:
        warnings.append(f"Paper protocol uses 800 points; manifest contains {points}")
    both_classes = all(
        counts["0"] > 0 and counts["1"] > 0 for counts in split_class_counts.values()
    )
    paper_ready = (
        complete
        and examples == 11_988
        and channels == 32
        and points == 800
        and both_classes
    )
    if require_complete_protocol and not paper_ready:
        raise ValueError(
            "Arrow dataset does not satisfy the complete SHU-MI paper protocol"
        )
    return SHUDatasetAudit(
        path=str(path),
        examples=examples,
        channels=channels,
        points=points,
        split_examples=split_examples,
        split_subjects=split_subjects,
        split_class_counts=split_class_counts,
        complete_subject_protocol=complete,
        paper_protocol_ready=paper_ready,
        expected_paper_examples=11_988,
        warnings=warnings,
    )


def summarize_manifest(manifest_path: str | Path) -> dict[str, object]:
    path = Path(manifest_path)
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    split_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"] or "unspecified")
        split_counts[split] = split_counts.get(split, 0) + 1
        shape = f"{int(row['num_channels'])}x{int(row['num_samples'])}"
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    return {
        "path": str(path),
        "examples": len(rows),
        "datasets": sorted({str(row["dataset_id"]) for row in rows}),
        "subjects": len({(row["dataset_id"], row["subject_id"]) for row in rows}),
        "tasks": sorted({str(row["task"]) for row in rows if row["task"] is not None}),
        "source_formats": sorted({str(row["source_format"]) for row in rows}),
        "split_examples": split_counts,
        "shape_examples": shape_counts,
        "labeled_examples": sum(row["label"] is not None for row in rows),
        "quality_flagged_examples": sum(bool(row["quality_flags"]) for row in rows),
        "shards": len({str(row["shard_path"]) for row in rows}),
    }
