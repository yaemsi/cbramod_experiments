from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from cbramod_experiments.data_harmonization import pipeline
from cbramod_experiments.data_harmonization.storage import HarmonizationSummary


def test_harmonize_edf_persists_lenient_source_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class FakeReader:
        def __init__(self, *, strict: bool) -> None:
            created["strict"] = strict

        def iter_windows(self, *args: Any, **kwargs: Any) -> Iterable[object]:
            return iter(())

        def audit_report(self) -> dict[str, Any]:
            return {
                "strict": False,
                "discovered_recordings": 2,
                "processed_recordings": 1,
                "skipped_recordings": 1,
                "failures": [
                    {
                        "path": "bad.edf",
                        "error_type": "ValueError",
                        "error": "malformed header",
                    }
                ],
            }

    class FakeWriter:
        def __init__(self, output_dir: str | Path, **_: Any) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def add_all(self, windows: Iterable[object]) -> None:
            list(windows)

        def close(self) -> HarmonizationSummary:
            manifest_path = self.output_dir / "manifest.parquet"
            manifest_path.touch()
            return HarmonizationSummary(
                output_dir=str(self.output_dir),
                manifest_path=str(manifest_path),
                examples=1,
                shards=1,
                datasets=["shu-mi"],
                source_formats=["edf"],
                split_examples={"train": 1},
                total_signal_bytes=128,
            )

    monkeypatch.setattr(pipeline, "SHUEdfReader", FakeReader)
    monkeypatch.setattr(pipeline, "ArrowShardWriter", FakeWriter)

    output_dir = tmp_path / "output"
    summary = pipeline.harmonize_shu_edf(
        tmp_path / "edf",
        output_dir,
        skip_invalid_recordings=True,
    )

    assert created["strict"] is False
    assert summary.source_audit is not None
    assert summary.source_audit["skipped_recordings"] == 1

    source_audit = json.loads((output_dir / "source_audit.json").read_text())
    assert source_audit["failures"][0]["path"] == "bad.edf"

    persisted_summary = json.loads((output_dir / "summary.json").read_text())
    assert persisted_summary["source_audit"]["skipped_recordings"] == 1
    assert persisted_summary["source_audit"]["report_path"].endswith(
        "source_audit.json"
    )
