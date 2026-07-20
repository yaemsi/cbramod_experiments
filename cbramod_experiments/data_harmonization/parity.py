from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..datasets.shumi import SHUH5Dataset
from .storage import ArrowEEGDataset


@dataclass(frozen=True)
class BackendParitySummary:
    compared_examples: int
    splits: list[str]
    max_absolute_difference: float
    labels_equal: bool
    shapes_equal: bool


def compare_hdf5_and_arrow(
    hdf5_path: str | Path,
    manifest_path: str | Path,
    *,
    max_examples_per_split: int | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> BackendParitySummary:
    compared = 0
    max_difference = 0.0
    labels_equal = True
    shapes_equal = True
    for split in ("train", "val", "test"):
        old = SHUH5Dataset(hdf5_path, split)
        new = ArrowEEGDataset(manifest_path, split)
        if len(old) != len(new):
            raise AssertionError(
                f"Length mismatch for {split}: HDF5={len(old)}, Arrow={len(new)}"
            )
        count = len(old)
        if max_examples_per_split is not None:
            count = min(count, max_examples_per_split)
        for index in range(count):
            old_signal, old_label = old[index]
            new_signal, new_label = new[index]
            if old_signal.shape != new_signal.shape:
                shapes_equal = False
                raise AssertionError(
                    f"Shape mismatch for {split}[{index}]: "
                    f"{old_signal.shape} vs {new_signal.shape}"
                )
            if int(old_label) != int(new_label):
                labels_equal = False
                raise AssertionError(
                    f"Label mismatch for {split}[{index}]: "
                    f"{int(old_label)} vs {int(new_label)}"
                )
            difference = float(torch.max(torch.abs(old_signal - new_signal)))
            max_difference = max(max_difference, difference)
            torch.testing.assert_close(
                old_signal,
                new_signal,
                rtol=rtol,
                atol=atol,
            )
            compared += 1
    return BackendParitySummary(
        compared_examples=compared,
        splits=["train", "val", "test"],
        max_absolute_difference=max_difference,
        labels_equal=labels_equal,
        shapes_equal=shapes_equal,
    )


def summary_dict(summary: BackendParitySummary) -> dict[str, object]:
    return asdict(summary)
