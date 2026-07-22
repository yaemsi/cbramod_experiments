from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shutil
import torch

from cbramod_experiments.data_harmonization import (
    ArrowBlockShuffleSampler,
    ArrowEEGDataset,
    EEGDataModule,
    audit_arrow_shu,
    compare_hdf5_and_arrow,
    harmonize_shu_mat,
)
from cbramod_experiments.datasets import preprocess_shu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources" / "data" / "shu-mi_dataset"

SAMPLE_MAT = RESOURCE_ROOT / "mat_files" / "sub-001_ses-01_task_motorimagery_eeg.mat"


def _single_session_mat_dir(tmp_path: Path) -> Path:
    if not SAMPLE_MAT.is_file():
        raise FileNotFoundError(f"Missing test fixture: {SAMPLE_MAT}")

    raw_dir = tmp_path / "mat_sample"
    raw_dir.mkdir()

    shutil.copy2(SAMPLE_MAT, raw_dir / SAMPLE_MAT.name)
    return raw_dir


def test_shu_mat_arrow_roundtrip_and_hdf5_parity(tmp_path: Path) -> None:
    raw_dir = _single_session_mat_dir(tmp_path)
    hdf5_path = tmp_path / "shu.h5"
    arrow_dir = tmp_path / "shu_arrow"

    preprocess_shu(
        raw_dir,
        hdf5_path,
        overwrite=True,
    )

    summary = harmonize_shu_mat(
        raw_dir,
        arrow_dir,
        records_per_batch=17,
        batches_per_shard=2,
        overwrite=True,
    )

    assert summary.examples == 100
    assert summary.shards == 3
    manifest_path = arrow_dir / "manifest.parquet"
    audit = audit_arrow_shu(manifest_path)
    assert audit.examples == 100
    assert audit.channels == 32
    assert audit.points == 800
    assert audit.split_examples == {"train": 100, "val": 0, "test": 0}

    parity = compare_hdf5_and_arrow(hdf5_path, manifest_path)
    assert parity.compared_examples == 100
    assert parity.max_absolute_difference == 0.0
    assert parity.labels_equal
    assert parity.shapes_equal

    dataset = ArrowEEGDataset(manifest_path, "train")
    signal, label = dataset[0]
    assert signal.shape == (32, 800)
    assert signal.dtype == torch.float32
    assert label.item() in {0, 1}

    manifest = pq.read_table(manifest_path).to_pylist()
    assert manifest[0]["sample_id"].startswith("shu-mi:sub-001:ses-01")
    assert manifest[0]["source_format"] == "mat"
    assert manifest[0]["channel_names"][0] == "Fp1"


def test_arrow_data_module_returns_training_batches(tmp_path: Path) -> None:
    output_dir = tmp_path / "shu_arrow"
    harmonize_shu_mat(
        _single_session_mat_dir(tmp_path),
        output_dir,
        records_per_batch=32,
        overwrite=True,
    )
    loaders = EEGDataModule(
        output_dir / "manifest.parquet",
        backend="arrow",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=7,
    ).loaders()
    signals, labels = next(iter(loaders["train"]))
    assert signals.shape == (8, 32, 800)
    assert labels.shape == (8,)
    assert np.isfinite(signals.numpy()).all()


def test_block_shuffle_sampler_visits_each_example_once(tmp_path: Path) -> None:
    output_dir = tmp_path / "shu_arrow"
    harmonize_shu_mat(
        RESOURCE_ROOT / "mat_files",
        output_dir,
        records_per_batch=10,
        batches_per_shard=2,
        overwrite=True,
    )
    dataset = ArrowEEGDataset(output_dir / "manifest.parquet", "train")
    sampler = ArrowBlockShuffleSampler(dataset, seed=11)
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert sorted(first) == list(range(len(dataset)))
    assert sorted(second) == list(range(len(dataset)))
    assert first != second
