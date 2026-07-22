from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def pytest_sessionstart() -> None:
    """Keep tiny CPU unit tests fast and deterministic on large CI machines."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture
def shu_real_root() -> Path:
    """Resolve an optional real SHU-MI root for integration tests."""
    candidates = []
    configured = os.environ.get("SHU_MI_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("resources/data/shu-mi_dataset"),
            Path("resources/shu-mi_dataset"),
        ]
    )
    for candidate in candidates:
        if (candidate / "mat_files").is_dir() and (candidate / "edf_files").is_dir():
            return candidate
    pytest.skip("Set SHU_MI_ROOT to a SHU-MI root containing mat_files and edf_files")
