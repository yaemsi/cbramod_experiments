from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import scipy.io

import cbramod_experiments.data_harmonization.parallel as parallel_module
from cbramod_experiments.data_harmonization import (
    ArrowEEGDataset,
    HarmonizationError,
    harmonize_shu_mat,
)
from cbramod_experiments.data_harmonization.parallel import (
    HarmonizationJobResult,
    _merge_job_outputs,
    estimate_recording_sizes,
    plan_recording_bundles,
    _prepare_output,
    _write_publication_plan,
)
from cbramod_experiments.data_harmonization.schema import EEGWindow
from cbramod_experiments.data_harmonization.storage import ArrowShardWriter


def _write_mat(
    path: Path,
    *,
    seed: int,
    trials: int = 3,
    valid: bool = True,
) -> None:
    generator = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {
        "data": generator.normal(size=(trials, 32, 40)).astype(np.float32),
    }
    if valid:
        payload["labels"] = np.asarray([[1, 2, 1]], dtype=np.int64)[:, :trials]
    scipy.io.savemat(path, payload)


def _make_source(root: Path, *, include_invalid: bool = False) -> Path:
    root.mkdir()
    _write_mat(
        root / "sub-001_ses-01_task_motorimagery_eeg.mat",
        seed=1,
    )
    _write_mat(
        root / "sub-002_ses-01_task_motorimagery_eeg.mat",
        seed=2,
    )
    if include_invalid:
        _write_mat(
            root / "sub-003_ses-01_task_motorimagery_eeg.mat",
            seed=3,
            valid=False,
        )
    return root


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def test_parallel_and_serial_harmonization_are_equivalent(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source")
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    serial = harmonize_shu_mat(
        source,
        serial_dir,
        target_sampling_rate_hz=250.0,
        original_sampling_rate_hz=250.0,
        amplitude_scale=1.0,
        records_per_batch=2,
        batches_per_shard=2,
        num_workers=1,
        overwrite=True,
        show_progress=False,
    )
    parallel = harmonize_shu_mat(
        source,
        parallel_dir,
        target_sampling_rate_hz=250.0,
        original_sampling_rate_hz=250.0,
        amplitude_scale=1.0,
        records_per_batch=2,
        batches_per_shard=2,
        num_workers=2,
        overwrite=True,
        show_progress=False,
    )

    assert serial.examples == parallel.examples == 6
    assert serial.split_examples == parallel.split_examples == {"train": 6}
    assert _manifest_rows(serial_dir / "manifest.parquet") == _manifest_rows(
        parallel_dir / "manifest.parquet"
    )

    serial_data = ArrowEEGDataset(serial_dir / "manifest.parquet", "train")
    parallel_data = ArrowEEGDataset(parallel_dir / "manifest.parquet", "train")
    assert len(serial_data) == len(parallel_data)
    for index in range(len(serial_data)):
        serial_signal, serial_label = serial_data[index]
        parallel_signal, parallel_label = parallel_data[index]
        np.testing.assert_array_equal(serial_signal.numpy(), parallel_signal.numpy())
        assert serial_label.item() == parallel_label.item()


def test_lenient_parallel_harmonization_records_failures(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source", include_invalid=True)
    output = tmp_path / "output"

    summary = harmonize_shu_mat(
        source,
        output,
        target_sampling_rate_hz=250.0,
        original_sampling_rate_hz=250.0,
        amplitude_scale=1.0,
        num_workers=2,
        overwrite=True,
        skip_invalid_recordings=True,
        show_progress=False,
    )

    assert summary.examples == 6
    assert summary.source_audit is not None
    assert summary.source_audit["discovered_recordings"] == 3
    assert summary.source_audit["processed_recordings"] == 2
    assert summary.source_audit["skipped_recordings"] == 1
    audit = json.loads((output / "source_audit.json").read_text(encoding="utf-8"))
    assert audit["failures"][0]["path"].endswith(
        "sub-003_ses-01_task_motorimagery_eeg.mat"
    )


def test_strict_parallel_harmonization_fails_and_keeps_work_dir(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path / "source", include_invalid=True)
    output = tmp_path / "output"

    with pytest.raises(HarmonizationError):
        harmonize_shu_mat(
            source,
            output,
            target_sampling_rate_hz=250.0,
            original_sampling_rate_hz=250.0,
            amplitude_scale=1.0,
            num_workers=2,
            overwrite=True,
            skip_invalid_recordings=False,
            show_progress=False,
        )

    assert (output / "source_audit.json").is_file()
    assert (output / "_work").is_dir()
    assert not (output / "manifest.parquet").exists()


def test_resume_reuses_completed_worker_outputs(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source", include_invalid=True)
    output = tmp_path / "output"

    with pytest.raises(HarmonizationError):
        harmonize_shu_mat(
            source,
            output,
            target_sampling_rate_hz=250.0,
            original_sampling_rate_hz=250.0,
            amplitude_scale=1.0,
            num_workers=1,
            overwrite=True,
            show_progress=False,
        )

    # Correct the invalid recording and reuse the two completed recording jobs.
    _write_mat(
        source / "sub-003_ses-01_task_motorimagery_eeg.mat",
        seed=3,
        valid=True,
    )
    summary = harmonize_shu_mat(
        source,
        output,
        target_sampling_rate_hz=250.0,
        original_sampling_rate_hz=250.0,
        amplitude_scale=1.0,
        num_workers=1,
        resume=True,
        show_progress=False,
    )

    assert summary.examples == 9
    assert summary.source_audit is not None
    assert summary.source_audit["resumed_recordings"] == 2
    assert not (output / "_work").exists()


def test_duplicate_shu_recordings_are_rejected_before_processing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    first = source / "copy-a"
    second = source / "copy-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    name = "sub-001_ses-01_task_motorimagery_eeg.mat"
    _write_mat(first / name, seed=1)
    _write_mat(second / name, seed=1)

    with pytest.raises(ValueError, match="same subject/session"):
        harmonize_shu_mat(
            source,
            tmp_path / "output",
            target_sampling_rate_hz=250.0,
            original_sampling_rate_hz=250.0,
            amplitude_scale=1.0,
            num_workers=2,
            overwrite=True,
            show_progress=False,
        )


def _worker_result(
    job_dir: Path,
    *,
    index: int,
    sample_id: str,
    source_uri: str,
) -> HarmonizationJobResult:
    signal = np.full((2, 8), float(index), dtype=np.float32)
    window = EEGWindow(
        signal=signal,
        sampling_rate_hz=4.0,
        channel_names=("C3", "C4"),
        channel_mask=np.ones(2, dtype=np.bool_),
        dataset_id="synthetic",
        subject_id=f"sub-{index:03d}",
        session_id="ses-01",
        task="merge-test",
        start_seconds=0.0,
        duration_seconds=2.0,
        label=index % 2,
        split="train",
        source_uri=source_uri,
        source_format="synthetic",
        sample_id=sample_id,
        amplitude_scale=1.0,
    )
    writer = ArrowShardWriter(
        job_dir,
        records_per_batch=1,
        batches_per_shard=1,
        overwrite=True,
    )
    writer.add(window)
    summary = writer.close()
    return HarmonizationJobResult(
        index=index,
        source_path=source_uri,
        job_dir=str(job_dir),
        manifest_path=summary.manifest_path,
        examples=summary.examples,
        shards=summary.shards,
        total_signal_bytes=summary.total_signal_bytes,
    )


def test_merge_validates_duplicate_ids_before_publishing(tmp_path: Path) -> None:
    output = tmp_path / "output"
    work = output / "_work"
    work.mkdir(parents=True)
    first = _worker_result(
        work / "job-000000",
        index=0,
        sample_id="duplicate-sample",
        source_uri="first.set",
    )
    second = _worker_result(
        work / "job-000001",
        index=1,
        sample_id="duplicate-sample",
        source_uri="second.set",
    )
    source_shards = list(work.glob("job-*/shards/*.arrow"))
    assert len(source_shards) == 2

    with pytest.raises(ValueError, match="Duplicate sample ID"):
        _merge_job_outputs(
            [first, second],
            output_dir=output,
            source_kind="bids",
            num_workers=2,
            processing_seconds=0.1,
            started_at=time.perf_counter(),
            skip_invalid_recordings=True,
            show_progress=False,
        )

    assert all(path.is_file() for path in source_shards)
    assert not (output / "shards").exists()
    assert not (output / "_publishing_shards").exists()
    assert not (output / "manifest.parquet").exists()


def test_merge_moves_shards_without_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    work = output / "_work"
    work.mkdir(parents=True)
    first = _worker_result(
        work / "job-000000",
        index=0,
        sample_id="sample-0",
        source_uri="run-1.set",
    )
    second = _worker_result(
        work / "job-000001",
        index=1,
        sample_id="sample-1",
        source_uri="run-2.set",
    )

    real_move = shutil.move
    move_calls: list[tuple[str, str]] = []

    def tracked_move(source: str, destination: str) -> str:
        move_calls.append((source, destination))
        return str(real_move(source, destination))

    def forbidden_copy(*_: object, **__: object) -> None:
        raise AssertionError("Final shard publication must not copy shard contents")

    monkeypatch.setattr(parallel_module.shutil, "move", tracked_move)
    monkeypatch.setattr(parallel_module.shutil, "copy2", forbidden_copy)

    summary = _merge_job_outputs(
        [second, first],
        output_dir=output,
        source_kind="bids",
        num_workers=2,
        processing_seconds=0.1,
        started_at=time.perf_counter(),
        skip_invalid_recordings=True,
        show_progress=False,
    )

    assert summary.examples == 2
    assert summary.shards == 2
    assert len(move_calls) == 2
    assert not (output / "_work").exists()
    assert not (output / "_publishing_shards").exists()
    rows = _manifest_rows(output / "manifest.parquet")
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1"]
    assert [row["shard_path"] for row in rows] == [
        "shards/shard-000000.arrow",
        "shards/shard-000001.arrow",
    ]
    assert all((output / str(row["shard_path"])).is_file() for row in rows)


def test_failed_shard_publication_rolls_back_worker_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    work = output / "_work"
    work.mkdir(parents=True)
    first = _worker_result(
        work / "job-000000",
        index=0,
        sample_id="sample-0",
        source_uri="run-1.set",
    )
    second = _worker_result(
        work / "job-000001",
        index=1,
        sample_id="sample-1",
        source_uri="run-2.set",
    )
    original_shards = list(work.glob("job-*/shards/*.arrow"))
    real_move = shutil.move
    forward_moves = 0

    def fail_second_publication(source: str, destination: str) -> str:
        nonlocal forward_moves
        if "_publishing_shards" in destination:
            forward_moves += 1
            if forward_moves == 2:
                raise OSError("simulated publication failure")
        return str(real_move(source, destination))

    monkeypatch.setattr(
        parallel_module.shutil,
        "move",
        fail_second_publication,
    )

    with pytest.raises(OSError, match="simulated publication failure"):
        _merge_job_outputs(
            [first, second],
            output_dir=output,
            source_kind="bids",
            num_workers=2,
            processing_seconds=0.1,
            started_at=time.perf_counter(),
            skip_invalid_recordings=True,
            show_progress=False,
        )

    assert all(path.is_file() for path in original_shards)
    assert not (output / "shards").exists()
    assert not (output / "_publishing_shards").exists()
    assert not (output / "manifest.parquet").exists()
    assert not (output / "summary.json").exists()


def test_resume_recovers_interrupted_move_publication(tmp_path: Path) -> None:
    output = tmp_path / "output"
    work = output / "_work"
    work.mkdir(parents=True)
    result = _worker_result(
        work / "job-000000",
        index=0,
        sample_id="sample-0",
        source_uri="run-1.set",
    )
    source_shard = next((Path(result.job_dir) / "shards").glob("*.arrow"))
    plan = [(source_shard, "shards/shard-000000.arrow")]
    _write_publication_plan(plan, output_dir=output)

    publishing = output / "_publishing_shards"
    publishing.mkdir()
    shutil.move(source_shard, publishing / "shard-000000.arrow")
    assert not source_shard.exists()

    returned_work = _prepare_output(output, overwrite=False, resume=True)

    assert returned_work == work
    assert source_shard.is_file()
    assert not publishing.exists()
    assert not (output / "_PUBLICATION_PLAN.json").exists()


def test_bundle_planner_is_deterministic_and_respects_limits(tmp_path: Path) -> None:
    paths = []
    for index, size in enumerate((90, 70, 40, 30, 20)):
        path = tmp_path / f"recording-{index}.set"
        path.write_bytes(b"x" * size)
        paths.append(path)

    first = plan_recording_bundles(
        paths,
        target_job_bytes=100,
        max_recordings_per_job=2,
    )
    second = plan_recording_bundles(
        list(reversed(paths)),
        target_job_bytes=100,
        max_recordings_per_job=2,
    )

    assert first == second
    assert sorted(path for bundle in first for path in bundle) == sorted(paths)
    assert all(len(bundle) <= 2 for bundle in first)
    assert all(sum(path.stat().st_size for path in bundle) <= 100 for bundle in first)


def test_bundled_worker_packs_multiple_recordings_into_a_shard(
    tmp_path: Path,
) -> None:
    source = _make_source(tmp_path / "source", include_invalid=False)
    output = tmp_path / "output"

    summary = harmonize_shu_mat(
        source,
        output,
        target_sampling_rate_hz=250.0,
        original_sampling_rate_hz=250.0,
        amplitude_scale=1.0,
        num_workers=2,
        target_job_gib=1.0,
        max_recordings_per_job=128,
        records_per_batch=32,
        batches_per_shard=16,
        overwrite=True,
        show_progress=False,
    )

    assert summary.examples == 6
    rows = pq.read_table(
        output / "manifest.parquet",
        columns=["shard_path", "source_uri"],
    ).to_pandas()
    sources_per_shard = rows.groupby("shard_path")["source_uri"].nunique()
    assert sources_per_shard.max() > 1
    assert summary.source_audit is not None
    assert summary.source_audit["jobs"] < summary.source_audit["discovered_recordings"]


def test_set_size_estimate_includes_sibling_fdt(tmp_path: Path) -> None:
    set_path = tmp_path / "sub-001_task-rest_eeg.set"
    fdt_path = set_path.with_suffix(".fdt")
    set_path.write_bytes(b"x" * 10)
    fdt_path.write_bytes(b"y" * 90)

    estimate = estimate_recording_sizes([set_path])[0]

    assert estimate.estimated_bytes == 100
