from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .readers import BIDSReader, SHUEdfReader, SHUMatReader
from .schema import DEFAULT_PREPROCESSING_VERSION
from .storage import ArrowShardWriter, HarmonizationSummary


def harmonize_shu_mat(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    target_sampling_rate_hz: float = 200.0,
    original_sampling_rate_hz: float = 250.0,
    amplitude_scale: float = 100.0,
    records_per_batch: int = 256,
    batches_per_shard: int = 16,
    overwrite: bool = False,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = SHUMatReader()
    windows = reader.iter_windows(
        raw_dir,
        target_sampling_rate_hz=target_sampling_rate_hz,
        original_sampling_rate_hz=original_sampling_rate_hz,
        amplitude_scale=amplitude_scale,
        preprocessing_version=preprocessing_version,
    )
    writer = ArrowShardWriter(
        output_dir,
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        overwrite=overwrite,
    )
    writer.add_all(windows)
    return writer.close()


def harmonize_shu_edf(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    events_root: str | Path | None = None,
    target_sampling_rate_hz: float | str = 200.0,
    amplitude_scale: float = 100.0,
    records_per_batch: int = 256,
    batches_per_shard: int = 16,
    overwrite: bool = False,
    skip_invalid_recordings: bool = False,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = SHUEdfReader(strict=not skip_invalid_recordings)
    windows = reader.iter_windows(
        raw_dir,
        events_root=events_root,
        target_sampling_rate_hz=target_sampling_rate_hz,
        amplitude_scale=amplitude_scale,
        preprocessing_version=preprocessing_version,
    )
    writer = ArrowShardWriter(
        output_dir,
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        overwrite=overwrite,
    )
    writer.add_all(windows)

    # Persist the source audit before finalizing the writer so that lenient runs
    # still explain what happened even when every discovered file was invalid.
    audit = reader.audit_report()
    output_path = Path(output_dir)
    audit_path = output_path / "source_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    audit["report_path"] = str(audit_path)

    summary = writer.close()
    summary = replace(summary, source_audit=audit)
    (output_path / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def harmonize_bids(
    root: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str = "hbn",
    target_sampling_rate_hz: float | None = None,
    window_seconds: float = 4.0,
    stride_seconds: float = 4.0,
    channel_policy: str = "preserve",
    channels: Sequence[str] | None = None,
    allow_missing_channels: bool = False,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    limit_recordings: int | None = None,
    amplitude_scale: float = 100.0,
    records_per_batch: int = 256,
    batches_per_shard: int = 16,
    overwrite: bool = False,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = BIDSReader(dataset_id=dataset_id)
    windows = reader.iter_windows(
        root,
        target_sampling_rate_hz=target_sampling_rate_hz,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        channel_policy=channel_policy,
        channels=channels,
        allow_missing_channels=allow_missing_channels,
        subjects=subjects,
        tasks=tasks,
        limit_recordings=limit_recordings,
        amplitude_scale=amplitude_scale,
        preprocessing_version=preprocessing_version,
    )
    writer = ArrowShardWriter(
        output_dir,
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        overwrite=overwrite,
    )
    writer.add_all(windows)
    return writer.close()
