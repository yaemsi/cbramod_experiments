from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from cbramod_experiments.datasets import preprocess_shu


def test_preprocess_shu_assigns_subject_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for subject in (1, 16, 21):
        savemat(
            raw / f"sub-{subject:03d}_session-1.mat",
            {
                "data": rng.normal(size=(2, 4, 1000)).astype(np.float32),
                "labels": np.array([[1, 2]], dtype=np.int64),
            },
        )

    output = tmp_path / "shu.h5"
    summary = preprocess_shu(raw, output)
    assert summary.examples == 6
    assert summary.target_points == 800

    with h5py.File(output, "r") as handle:
        assert handle["signals"].shape == (6, 4, 800)
        assert handle["splits/train"].shape == (2,)
        assert handle["splits/val"].shape == (2,)
        assert handle["splits/test"].shape == (2,)
        assert handle["labels"][:].tolist() == [0, 1, 0, 1, 0, 1]


def test_audit_rejects_incomplete_protocol_in_strict_mode(tmp_path: Path) -> None:
    import pytest

    from cbramod_experiments.datasets import audit_shu_h5

    raw = tmp_path / "raw_partial"
    raw.mkdir()
    savemat(
        raw / "sub-001_ses-01_task_motorimagery_eeg.mat",
        {
            "data": np.zeros((2, 32, 1000), dtype=np.float32),
            "labels": np.array([[1, 2]], dtype=np.int64),
        },
    )
    output = tmp_path / "partial.h5"
    preprocess_shu(raw, output)
    audit = audit_shu_h5(output)
    assert audit.complete_subject_protocol is False
    assert audit.split_examples == {"train": 2, "val": 0, "test": 0}
    with pytest.raises(ValueError, match="complete paper protocol"):
        audit_shu_h5(output, require_complete_protocol=True)
