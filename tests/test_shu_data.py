from pathlib import Path

import h5py
import numpy as np

from cbramod_experiments.datasets import SHUH5Dataset, parse_subject_id, subject_split


def test_subject_protocol() -> None:
    assert subject_split(1) == "train"
    assert subject_split(15) == "train"
    assert subject_split(16) == "val"
    assert subject_split(20) == "val"
    assert subject_split(21) == "test"
    assert subject_split(25) == "test"


def test_subject_parser() -> None:
    assert parse_subject_id("sub-001_session-2.mat") == 1
    assert parse_subject_id("subject20_sess5.mat") == 20


def test_h5_dataset(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "signals", data=np.ones((2, 32, 800), dtype=np.float32) * 100
        )
        handle.create_dataset("labels", data=np.array([0, 1], dtype=np.int8))
        splits = handle.create_group("splits")
        splits.create_dataset("train", data=np.array([0, 1], dtype=np.int64))
        splits.create_dataset("val", data=np.array([], dtype=np.int64))
        splits.create_dataset("test", data=np.array([], dtype=np.int64))
        handle.attrs["amplitude_scale"] = 100.0
    dataset = SHUH5Dataset(path, "train")
    signal, label = dataset[1]
    assert signal.shape == (32, 800)
    assert float(signal.mean()) == 1.0
    assert int(label) == 1
