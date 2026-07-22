from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .parallel import harmonize_recordings
from .readers import BIDSReader, SHUEdfReader, SHUMatReader
from .schema import DEFAULT_PREPROCESSING_VERSION
from .storage import HarmonizationSummary


def harmonize_shu_mat(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    target_sampling_rate_hz: float = 200.0,
    original_sampling_rate_hz: float = 250.0,
    amplitude_scale: float = 100.0,
    records_per_batch: int = 256,
    batches_per_shard: int = 16,
    num_workers: int = 1,
    target_job_gib: float = 0.0,
    max_recordings_per_job: int = 128,
    overwrite: bool = False,
    resume: bool = False,
    skip_invalid_recordings: bool = False,
    show_progress: bool = True,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = SHUMatReader()
    files = reader.discover(raw_dir)
    return harmonize_recordings(
        source_kind="shu-mat",
        source_paths=files,
        dataset_root=raw_dir,
        output_dir=output_dir,
        reader_options={
            "target_sampling_rate_hz": target_sampling_rate_hz,
            "original_sampling_rate_hz": original_sampling_rate_hz,
            "amplitude_scale": amplitude_scale,
            "preprocessing_version": preprocessing_version,
        },
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        num_workers=num_workers,
        target_job_bytes=int(target_job_gib * 1024**3),
        max_recordings_per_job=max_recordings_per_job,
        overwrite=overwrite,
        resume=resume,
        skip_invalid_recordings=skip_invalid_recordings,
        show_progress=show_progress,
    )


def harmonize_shu_edf(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    events_root: str | Path | None = None,
    target_sampling_rate_hz: float | str = 200.0,
    amplitude_scale: float = 100.0,
    records_per_batch: int = 256,
    batches_per_shard: int = 16,
    num_workers: int = 1,
    target_job_gib: float = 0.0,
    max_recordings_per_job: int = 128,
    overwrite: bool = False,
    resume: bool = False,
    skip_invalid_recordings: bool = False,
    show_progress: bool = True,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = SHUEdfReader(strict=True)
    files = reader.discover(raw_dir)
    return harmonize_recordings(
        source_kind="shu-edf",
        source_paths=files,
        dataset_root=raw_dir,
        output_dir=output_dir,
        reader_options={
            "events_root": str(events_root)
            if events_root is not None
            else str(raw_dir),
            "target_sampling_rate_hz": target_sampling_rate_hz,
            "amplitude_scale": amplitude_scale,
            "preprocessing_version": preprocessing_version,
        },
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        num_workers=num_workers,
        target_job_bytes=int(target_job_gib * 1024**3),
        max_recordings_per_job=max_recordings_per_job,
        overwrite=overwrite,
        resume=resume,
        skip_invalid_recordings=skip_invalid_recordings,
        show_progress=show_progress,
    )


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
    num_workers: int = 1,
    target_job_gib: float = 0.0,
    max_recordings_per_job: int = 128,
    overwrite: bool = False,
    resume: bool = False,
    skip_invalid_recordings: bool = False,
    show_progress: bool = True,
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
) -> HarmonizationSummary:
    reader = BIDSReader(dataset_id=dataset_id)
    files = reader.discover(
        root,
        subjects=subjects,
        tasks=tasks,
        limit_recordings=limit_recordings,
    )
    return harmonize_recordings(
        source_kind="bids",
        source_paths=files,
        dataset_root=root,
        output_dir=output_dir,
        reader_options={
            "dataset_id": dataset_id,
            "target_sampling_rate_hz": target_sampling_rate_hz,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "channel_policy": channel_policy,
            "channels": tuple(channels) if channels is not None else None,
            "allow_missing_channels": allow_missing_channels,
            "amplitude_scale": amplitude_scale,
            "preprocessing_version": preprocessing_version,
        },
        records_per_batch=records_per_batch,
        batches_per_shard=batches_per_shard,
        num_workers=num_workers,
        target_job_bytes=int(target_job_gib * 1024**3),
        max_recordings_per_job=max_recordings_per_job,
        overwrite=overwrite,
        resume=resume,
        skip_invalid_recordings=skip_invalid_recordings,
        show_progress=show_progress,
    )
