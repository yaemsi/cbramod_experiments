from __future__ import annotations

import importlib


def test_public_packages_import_without_cycles() -> None:
    data_harmonization = importlib.import_module(
        "cbramod_experiments.data_harmonization"
    )
    datasets = importlib.import_module("cbramod_experiments.datasets")
    models = importlib.import_module("cbramod_experiments.models")
    utils = importlib.import_module("cbramod_experiments.utils")

    assert data_harmonization.EEGDataModule is not None
    assert data_harmonization.ArrowEEGDataset is not None
    assert datasets.SHUDataModule is not None
    assert datasets.parse_subject_id is not None
    assert models.build_model is not None
    assert utils.ModelConfig is not None
