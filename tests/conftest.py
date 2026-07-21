from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart() -> None:
    """Keep tiny CPU unit tests fast and deterministic on large CI machines."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture(scope="session")
def synthetic_shu_mat_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create one deterministic, 100-trial SHU-MI-like MATLAB session.

    MAT/Arrow/HDF5 unit tests must not accidentally scan the complete 11,988-
    example local dataset.  This fixture preserves the real SHU tensor and label
    contract while remaining self-contained and fast enough for CI.
    """
    root = tmp_path_factory.mktemp("synthetic_shu")
    mat_dir = root / "mat_files"
    mat_dir.mkdir(parents=True)

    rng = np.random.default_rng(1234)
    data = rng.normal(0.0, 10.0, size=(100, 32, 1000)).astype(np.float32)
    labels = (np.arange(100, dtype=np.int64) % 2 + 1)[None, :]
    scipy.io.savemat(
        mat_dir / "sub-001_ses-01_task_motorimagery_eeg.mat",
        {"data": data, "labels": labels},
    )
    return root


def _resolve_shu_data_root() -> Path:
    configured = os.environ.get("SHU_MI_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / "resources" / "data" / "shu-mi_dataset",
        PROJECT_ROOT / "resources" / "shu-mi_dataset",  # legacy layout
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates if path is not None)
    pytest.skip(
        "Real SHU-MI data is required for this integration test. "
        f"Set SHU_MI_ROOT or place it at the project default. Searched: {searched}"
    )


def _find_required(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        pytest.skip(f"Required SHU-MI integration file is missing: {filename}")
    return matches[0]


@pytest.fixture(scope="session")
def shu_single_session_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage exactly one real MAT/EDF/events session from the full archive.

    The integration tests intentionally operate on 100 trials, even when the
    configured source root contains the full 11,988-example SHU-MI dataset.
    """
    source_root = _resolve_shu_data_root()
    staged_root = tmp_path_factory.mktemp("shu_single_session")

    files = {
        "mat_files": "sub-001_ses-01_task_motorimagery_eeg.mat",
        "edf_files": "sub-001_ses-01_task_motorimagery_eeg.edf",
        "events": "sub-001_ses-01_task_motorimagery_events.tsv",
    }
    for directory, filename in files.items():
        destination = staged_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_find_required(source_root, filename), destination / filename)

    return staged_root
