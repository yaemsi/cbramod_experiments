from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cbramod_experiments.data_harmonization import pipeline
from cbramod_experiments.data_harmonization.storage import (
    HarmonizationSummary,
)


def test_harmonize_edf_forwards_lenient_mode_to_shared_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeReader:
        def __init__(self, *, strict: bool) -> None:
            captured["reader_strict"] = strict

        def discover(self, root: str | Path) -> list[Path]:
            root = Path(root)
            return [
                root / "sub-001_ses-01_task_motorimagery_eeg.edf",
                root / "sub-002_ses-01_task_motorimagery_eeg.edf",
            ]

    def fake_harmonize_recordings(
        **kwargs: Any,
    ) -> HarmonizationSummary:
        captured.update(kwargs)

        output_dir = Path(kwargs["output_dir"])
        audit = {
            "source_kind": "shu-edf",
            "strict": False,
            "num_workers": kwargs["num_workers"],
            "discovered_recordings": 2,
            "processed_recordings": 1,
            "skipped_recordings": 1,
            "resumed_recordings": 0,
            "failures": [
                {
                    "path": "bad.edf",
                    "error_type": "ValueError",
                    "error": "malformed header",
                }
            ],
        }

        return HarmonizationSummary(
            output_dir=str(output_dir),
            manifest_path=str(output_dir / "manifest.parquet"),
            examples=100,
            shards=1,
            datasets=["shu-mi"],
            source_formats=["edf"],
            split_examples={"train": 100},
            total_signal_bytes=128,
            source_audit=audit,
        )

    monkeypatch.setattr(
        pipeline,
        "SHUEdfReader",
        FakeReader,
    )
    monkeypatch.setattr(
        pipeline,
        "harmonize_recordings",
        fake_harmonize_recordings,
    )

    edf_root = tmp_path / "edf"
    events_root = tmp_path / "events"
    output_dir = tmp_path / "output"

    summary = pipeline.harmonize_shu_edf(
        edf_root,
        output_dir,
        events_root=events_root,
        target_sampling_rate_hz=200,
        num_workers=3,
        skip_invalid_recordings=True,
        show_progress=False,
    )

    # Individual worker readers remain strict. The shared coordinator decides
    # whether recording-level failures abort or are recorded and skipped.
    assert captured["reader_strict"] is True

    assert captured["source_kind"] == "shu-edf"
    assert captured["source_paths"] == [
        edf_root / "sub-001_ses-01_task_motorimagery_eeg.edf",
        edf_root / "sub-002_ses-01_task_motorimagery_eeg.edf",
    ]
    assert captured["dataset_root"] == edf_root
    assert captured["output_dir"] == output_dir
    assert captured["num_workers"] == 3
    assert captured["skip_invalid_recordings"] is True
    assert captured["show_progress"] is False

    reader_options = captured["reader_options"]
    assert reader_options["events_root"] == str(events_root)
    assert reader_options["target_sampling_rate_hz"] == 200

    assert summary.source_audit is not None
    assert summary.source_audit["skipped_recordings"] == 1
    assert summary.source_audit["failures"][0]["path"] == "bad.edf"
