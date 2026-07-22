from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cbramod_experiments.data_harmonization import BIDSReader, harmonize_bids
from cbramod_experiments.data_harmonization.readers import (
    bids_recording_id,
    parse_bids_entities,
)
from cbramod_experiments.data_harmonization.schema import EEGRecording
from cbramod_experiments.data_harmonization.storage import ArrowEEGDataset
from cbramod_experiments.data_harmonization.transforms import sliding_windows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources" / "data" / "shu-mi"


def test_parse_bids_entities() -> None:
    entities = parse_bids_entities(
        "sub-NDARAA123_ses-HBNsiteRU_task-RestingState_eeg.bdf"
    )
    assert entities == {
        "sub": "NDARAA123",
        "ses": "HBNsiteRU",
        "task": "RestingState",
    }


def test_bids_recording_identity_preserves_run_entities() -> None:
    run_1 = "sub-ABC_task-rest_run-1_eeg.set"
    run_2 = "sub-ABC_task-rest_run-2_eeg.set"
    assert parse_bids_entities(run_1)["run"] == "1"
    assert parse_bids_entities(run_2)["run"] == "2"
    assert bids_recording_id(run_1) == "sub-ABC_task-rest_run-1"
    assert bids_recording_id(run_2) == "sub-ABC_task-rest_run-2"


def test_bids_runs_generate_disjoint_window_ids() -> None:
    def recording(run: int) -> EEGRecording:
        recording_id = f"sub-ABC_task-rest_run-{run}"
        return EEGRecording(
            signal=np.zeros((2, 16), dtype=np.float32),
            sampling_rate_hz=4.0,
            channel_names=("C3", "C4"),
            dataset_id="hbn",
            subject_id="sub-ABC",
            session_id=None,
            task="task-rest",
            source_uri=f"{recording_id}_eeg.set",
            source_format="set",
            metadata={"recording_id": recording_id, "run_id": str(run)},
        )

    ids_1 = {
        window.sample_id
        for window in sliding_windows(
            recording(1),
            window_seconds=2.0,
            stride_seconds=2.0,
            split=None,
            amplitude_scale=1.0,
            preprocessing_version="test",
        )
    }
    ids_2 = {
        window.sample_id
        for window in sliding_windows(
            recording(2),
            window_seconds=2.0,
            stride_seconds=2.0,
            split=None,
            amplitude_scale=1.0,
            preprocessing_version="test",
        )
    }
    assert ids_1.isdisjoint(ids_2)


@pytest.mark.integration
def test_generic_bids_reader_supports_edf_and_arrow_materialization(
    tmp_path: Path,
) -> None:
    source = RESOURCE_ROOT / "edf_files" / "sub-001_ses-01_task_motorimagery_eeg.edf"
    bids_root = tmp_path / "bids"
    eeg_dir = bids_root / "sub-001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    target = eeg_dir / source.name
    target.write_bytes(source.read_bytes())

    reader = BIDSReader(dataset_id="poc-bids")
    recordings = reader.discover(bids_root)
    assert recordings == [target]
    recording = reader.read_recording(target, root=bids_root)
    assert recording.signal.shape == (32, 100_000)
    assert recording.sampling_rate_hz == 250.0
    assert recording.subject_id == "sub-001"

    output_dir = tmp_path / "harmonized"
    summary = harmonize_bids(
        bids_root,
        output_dir,
        dataset_id="poc-bids",
        target_sampling_rate_hz=200.0,
        window_seconds=4.0,
        stride_seconds=4.0,
        limit_recordings=1,
        records_per_batch=40,
        overwrite=True,
    )
    assert summary.examples == 100
    # HBN-style generic windows are intentionally unlabeled and therefore are
    # inspected with require_labels=False rather than used in SHU supervision.
    dataset = ArrowEEGDataset(
        output_dir / "manifest.parquet", None, require_labels=False
    )
    assert len(dataset) == 100
    signal, label = dataset[0]
    assert signal.shape == (32, 800)
    assert label.item() == -1
