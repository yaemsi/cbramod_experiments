from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .readers import BIDSReader, SHUEdfReader, SHUMatReader
from .schema import EEGWindow
from .storage import ArrowShardWriter, HarmonizationSummary

SourceKind = Literal["shu-mat", "shu-edf", "bids"]

_EXPECTED_SOURCE_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
)


@dataclass(frozen=True)
class RecordingEstimate:
    """Deterministic size estimate used to group source recordings into jobs."""

    path: str
    estimated_bytes: int


@dataclass(frozen=True)
class HarmonizationJob:
    """One independently processable bundle of EEG recordings."""

    index: int
    source_kind: SourceKind
    source_paths: tuple[str, ...]
    dataset_root: str
    work_dir: str
    reader_options: dict[str, Any]
    records_per_batch: int
    batches_per_shard: int


@dataclass(frozen=True)
class HarmonizationJobResult:
    """Small, picklable result returned by one worker process."""

    index: int
    source_path: str
    job_dir: str
    manifest_path: str | None
    examples: int
    shards: int
    total_signal_bytes: int
    failure: dict[str, str] | None = None
    failures: tuple[dict[str, str], ...] = ()
    attempted_recordings: int = 1
    processed_recordings: int = 1
    resumed: bool = False


def estimate_recording_sizes(
    source_paths: Sequence[str | Path],
) -> list[RecordingEstimate]:
    """Estimate work volume from source-file bytes without loading recordings.

    Source size is a cheap, format-independent proxy for output volume. It is
    sufficient for balancing MAT, EDF, SET/FDT, and BDF recordings into a small
    number of deterministic worker bundles.
    """

    estimates: list[RecordingEstimate] = []
    for source_path in source_paths:
        path = Path(source_path)
        estimated_bytes = path.stat().st_size
        if path.suffix.casefold() == ".set":
            # EEGLAB commonly stores samples in a sibling .fdt file.
            fdt_path = path.with_suffix(".fdt")
            if fdt_path.is_file():
                estimated_bytes += fdt_path.stat().st_size
        estimates.append(
            RecordingEstimate(
                path=str(path),
                estimated_bytes=max(1, estimated_bytes),
            )
        )
    return estimates


def plan_recording_bundles(
    source_paths: Sequence[str | Path],
    *,
    target_job_bytes: int,
    max_recordings_per_job: int,
) -> list[tuple[Path, ...]]:
    """Pack recordings deterministically using first-fit decreasing.

    ``target_job_bytes <= 0`` preserves the legacy one-recording-per-job path.
    """

    if max_recordings_per_job <= 0:
        raise ValueError("max_recordings_per_job must be positive")

    ordered_paths = [Path(path) for path in source_paths]
    if target_job_bytes <= 0:
        return [(path,) for path in ordered_paths]

    estimates = estimate_recording_sizes(ordered_paths)
    estimates.sort(key=lambda item: (-item.estimated_bytes, item.path))

    bundles: list[list[RecordingEstimate]] = []
    bundle_bytes: list[int] = []
    for estimate in estimates:
        selected: int | None = None
        for index, bundle in enumerate(bundles):
            if (
                len(bundle) < max_recordings_per_job
                and bundle_bytes[index] + estimate.estimated_bytes <= target_job_bytes
            ):
                selected = index
                break
        if selected is None:
            bundles.append([estimate])
            bundle_bytes.append(estimate.estimated_bytes)
        else:
            bundles[selected].append(estimate)
            bundle_bytes[selected] += estimate.estimated_bytes

    planned: list[tuple[Path, ...]] = []
    for bundle in bundles:
        planned.append(
            tuple(
                Path(item.path) for item in sorted(bundle, key=lambda item: item.path)
            )
        )
    planned.sort(key=lambda paths: tuple(str(path) for path in paths))
    return planned


class HarmonizationError(RuntimeError):
    """Raised when strict harmonization encounters invalid source recordings."""

    def __init__(self, failures: Sequence[dict[str, str]]) -> None:
        self.failures = list(failures)
        preview = "\n".join(
            f"- {item['path']} ({item['error_type']}): {item['error']}"
            for item in self.failures[:10]
        )
        suffix = "" if len(self.failures) <= 10 else "\n- ..."
        super().__init__(
            f"Harmonization failed for {len(self.failures)} recording(s):\n"
            f"{preview}{suffix}"
        )


def _rank_zero() -> bool:
    """Return whether this process should own terminal progress output."""

    value = os.environ.get("RANK")
    return value in {None, "", "0"}


def _success_marker(job_dir: Path) -> Path:
    return job_dir / "_SUCCESS.json"


def _job_fingerprint(job: HarmonizationJob) -> str:
    """Fingerprint all source files and preprocessing settings for safe resume."""

    sources = []
    for source_path in job.source_paths:
        source = Path(source_path)
        stat = source.stat()
        sources.append(
            {
                "path": str(source.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "source_kind": job.source_kind,
        "sources": sources,
        "dataset_root": str(Path(job.dataset_root).resolve()),
        "reader_options": job.reader_options,
        "records_per_batch": job.records_per_batch,
        "batches_per_shard": job.batches_per_shard,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_completed_job(job: HarmonizationJob) -> HarmonizationJobResult | None:
    job_dir = Path(job.work_dir)
    marker = _success_marker(job_dir)
    manifest = job_dir / "manifest.parquet"
    summary_path = job_dir / "summary.json"
    if not (marker.is_file() and manifest.is_file() and summary_path.is_file()):
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("source_paths") != list(job.source_paths):
        return None
    if payload.get("job_fingerprint") != _job_fingerprint(job):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures = tuple(payload.get("failures", ()))
    processed = int(payload.get("processed_recordings", len(job.source_paths)))
    return HarmonizationJobResult(
        index=job.index,
        source_path=job.source_paths[0],
        job_dir=str(job_dir),
        manifest_path=str(manifest),
        examples=int(summary["examples"]),
        shards=int(summary["shards"]),
        total_signal_bytes=int(summary["total_signal_bytes"]),
        failure=failures[0] if failures else None,
        failures=failures,
        attempted_recordings=len(job.source_paths),
        processed_recordings=processed,
        resumed=True,
    )


def _job_windows(
    job: HarmonizationJob,
    source_path: str | Path,
) -> Iterable[EEGWindow]:
    options = dict(job.reader_options)
    source = Path(source_path)
    dataset_root = Path(job.dataset_root)

    if job.source_kind == "shu-mat":
        reader = SHUMatReader()
        return reader.iter_windows(source, **options)

    if job.source_kind == "shu-edf":
        reader = SHUEdfReader(strict=True)
        return reader.iter_windows(source, **options)

    if job.source_kind == "bids":
        dataset_id = str(options.pop("dataset_id"))
        reader = BIDSReader(dataset_id=dataset_id)
        return reader.iter_windows(
            source,
            metadata_root=dataset_root,
            **options,
        )

    raise ValueError(f"Unsupported source kind: {job.source_kind}")


def run_harmonization_job(job: HarmonizationJob) -> HarmonizationJobResult:
    """Process a recording bundle into packed worker-local Arrow shards."""

    completed = _load_completed_job(job)
    if completed is not None:
        return completed

    job_dir = Path(job.work_dir)
    failures: list[dict[str, str]] = []
    processed_recordings = 0
    writer: ArrowShardWriter | None = None

    for source_path in job.source_paths:
        try:
            # Materialize one source atomically so a malformed recording
            # cannot leave partial windows in an otherwise valid bundle.
            windows = list(_job_windows(job, source_path))
            if not windows:
                raise ValueError("Recording produced no EEG windows")
            if writer is None:
                writer = ArrowShardWriter(
                    job_dir,
                    records_per_batch=job.records_per_batch,
                    batches_per_shard=job.batches_per_shard,
                    overwrite=True,
                )
            writer.add_all(windows)
            processed_recordings += 1
        except _EXPECTED_SOURCE_ERRORS as exc:
            failures.append(
                {
                    "path": str(source_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if writer is None:
        shutil.rmtree(job_dir, ignore_errors=True)
        return HarmonizationJobResult(
            index=job.index,
            source_path=job.source_paths[0],
            job_dir=str(job_dir),
            manifest_path=None,
            examples=0,
            shards=0,
            total_signal_bytes=0,
            failure=failures[0] if failures else None,
            failures=tuple(failures),
            attempted_recordings=len(job.source_paths),
            processed_recordings=0,
        )

    summary = writer.close()
    _success_marker(job_dir).write_text(
        json.dumps(
            {
                "source_paths": list(job.source_paths),
                "source_kind": job.source_kind,
                "job_index": job.index,
                "job_fingerprint": _job_fingerprint(job),
                "processed_recordings": processed_recordings,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return HarmonizationJobResult(
        index=job.index,
        source_path=job.source_paths[0],
        job_dir=str(job_dir),
        manifest_path=str(job_dir / "manifest.parquet"),
        examples=summary.examples,
        shards=summary.shards,
        total_signal_bytes=summary.total_signal_bytes,
        failure=failures[0] if failures else None,
        failures=tuple(failures),
        attempted_recordings=len(job.source_paths),
        processed_recordings=processed_recordings,
    )


def _prepare_output(
    output_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be enabled")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif resume:
            # Preserve completed worker outputs and recover any interrupted
            # move-based publication before removing partial final metadata.
            _recover_interrupted_publication(output_dir)
            shutil.rmtree(output_dir / "shards", ignore_errors=True)
            shutil.rmtree(output_dir / "_publishing_shards", ignore_errors=True)
            for name in ("manifest.parquet", "summary.json", "source_audit.json"):
                (output_dir / name).unlink(missing_ok=True)
                _temporary_path(output_dir / name).unlink(missing_ok=True)
        else:
            raise FileExistsError(
                f"Output exists: {output_dir}; pass overwrite=True or resume=True"
            )

    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _collect_results(
    jobs: Sequence[HarmonizationJob],
    *,
    num_workers: int,
    show_progress: bool,
    description: str,
) -> list[HarmonizationJobResult]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    progress_enabled = show_progress and _rank_zero()
    results: list[HarmonizationJobResult] = []
    processed_examples = 0
    skipped = 0

    with tqdm(
        total=sum(len(job.source_paths) for job in jobs),
        desc=description,
        unit="recording",
        dynamic_ncols=True,
        disable=not progress_enabled,
    ) as progress:
        if num_workers == 1:
            for job in jobs:
                result = run_harmonization_job(job)
                results.append(result)
                processed_examples += result.examples
                skipped += len(result.failures)
                progress.update(result.attempted_recordings)
                progress.set_postfix(examples=processed_examples, skipped=skipped)
            return results

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=context,
        ) as executor:
            futures: dict[Future[HarmonizationJobResult], int] = {
                executor.submit(run_harmonization_job, job): job.index for job in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                processed_examples += result.examples
                skipped += len(result.failures)
                progress.update(result.attempted_recordings)
                progress.set_postfix(examples=processed_examples, skipped=skipped)

    return results


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _publication_plan_path(output_dir: Path) -> Path:
    return output_dir / "_PUBLICATION_PLAN.json"


def _write_publication_plan(
    shard_plan: Sequence[tuple[Path, str]],
    *,
    output_dir: Path,
) -> None:
    _atomic_write_json(
        _publication_plan_path(output_dir),
        {
            "shards": [
                {
                    "source": str(source.resolve()),
                    "final_relative": final_relative,
                }
                for source, final_relative in shard_plan
            ]
        },
    )


def _recover_interrupted_publication(output_dir: Path) -> None:
    """Restore shards moved before an interrupted coordinator finalization."""

    plan_path = _publication_plan_path(output_dir)
    if not plan_path.is_file():
        publishing_dir = output_dir / "_publishing_shards"
        if publishing_dir.exists() and any(publishing_dir.iterdir()):
            raise RuntimeError(
                "Found staged shards without a publication plan; refusing to "
                f"delete potentially valid data in {publishing_dir}"
            )
        shutil.rmtree(publishing_dir, ignore_errors=True)
        return

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = payload.get("shards")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid publication plan: {plan_path}")
    shard_plan: list[tuple[Path, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid publication plan row: {row!r}")
        source = row.get("source")
        final_relative = row.get("final_relative")
        if not isinstance(source, str) or not isinstance(final_relative, str):
            raise ValueError(f"Invalid publication plan row: {row!r}")
        shard_plan.append((Path(source), final_relative))

    _rollback_published_shards(shard_plan, output_dir=output_dir)
    plan_path.unlink(missing_ok=True)


def _rollback_published_shards(
    shard_plan: Sequence[tuple[Path, str]],
    *,
    output_dir: Path,
) -> None:
    """Return published shards to worker directories after a failed finalization."""

    publishing_dir = output_dir / "_publishing_shards"
    final_shards_dir = output_dir / "shards"

    for source, final_relative in reversed(shard_plan):
        final_name = Path(final_relative).name
        candidates = (
            final_shards_dir / final_name,
            publishing_dir / final_name,
        )
        current = next((path for path in candidates if path.exists()), None)
        if current is None:
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current), str(source))

    shutil.rmtree(publishing_dir, ignore_errors=True)
    shutil.rmtree(final_shards_dir, ignore_errors=True)


def _publish_shards(
    shard_plan: Sequence[tuple[Path, str]],
    *,
    output_dir: Path,
    show_progress: bool,
) -> None:
    """Move validated worker shards into a temporary publication directory.

    ``shutil.move`` resolves to a filesystem rename when source and destination
    are on the same filesystem, avoiding a second full copy of the corpus. The
    temporary directory is renamed to ``shards/`` only after every move
    succeeds. On failure, already moved shards are restored to their worker
    directories so the run remains resumable.
    """

    publishing_dir = output_dir / "_publishing_shards"
    final_shards_dir = output_dir / "shards"
    shutil.rmtree(publishing_dir, ignore_errors=True)
    publishing_dir.mkdir(parents=True, exist_ok=True)

    progress_enabled = show_progress and _rank_zero()
    try:
        with tqdm(
            shard_plan,
            desc="Publishing Arrow shards",
            unit="shard",
            dynamic_ncols=True,
            disable=not progress_enabled,
        ) as progress:
            for source, final_relative in progress:
                destination = publishing_dir / Path(final_relative).name
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(source), str(destination))

        if final_shards_dir.exists():
            shutil.rmtree(final_shards_dir)
        publishing_dir.replace(final_shards_dir)
    except Exception:
        _rollback_published_shards(shard_plan, output_dir=output_dir)
        _publication_plan_path(output_dir).unlink(missing_ok=True)
        raise


def _merge_job_outputs(
    results: Sequence[HarmonizationJobResult],
    *,
    output_dir: Path,
    source_kind: SourceKind,
    num_workers: int,
    processing_seconds: float,
    started_at: float,
    skip_invalid_recordings: bool,
    show_progress: bool,
) -> HarmonizationSummary:
    ordered = sorted(results, key=lambda result: result.index)
    failure_rows = [
        failure
        for result in ordered
        for failure in (
            result.failures
            if result.failures
            else ((result.failure,) if result.failure is not None else ())
        )
    ]
    successful = [result for result in ordered if result.manifest_path is not None]

    audit_path = output_dir / "source_audit.json"
    discovered_recordings = sum(result.attempted_recordings for result in ordered)
    processed_recordings = sum(result.processed_recordings for result in ordered)
    resumed_recordings = sum(
        result.processed_recordings for result in successful if result.resumed
    )
    audit: dict[str, Any] = {
        "source_kind": source_kind,
        "strict": not skip_invalid_recordings,
        "num_workers": num_workers,
        "jobs": len(ordered),
        "discovered_recordings": discovered_recordings,
        "processed_recordings": processed_recordings,
        "skipped_recordings": len(failure_rows),
        "resumed_recordings": resumed_recordings,
        "failures": failure_rows,
        "report_path": str(audit_path),
    }
    _atomic_write_json(audit_path, audit)

    if failure_rows and not skip_invalid_recordings:
        raise HarmonizationError(failure_rows)
    if not successful:
        raise ValueError("No valid EEG recordings were harmonized")

    progress_enabled = show_progress and _rank_zero()
    if progress_enabled:
        print("Validating worker manifests and sample IDs...", flush=True)

    # Pass 1: validate every fragment and build a publication plan. No shard is
    # moved until schemas, source paths, and all sample IDs have been checked.
    merged_rows: list[dict[str, Any]] = []
    manifest_schema: pa.Schema | None = None
    sample_sources: dict[str, str] = {}
    shard_plan: list[tuple[Path, str]] = []
    shard_index = 0

    for result in successful:
        if result.manifest_path is None:
            raise RuntimeError(f"Successful job has no manifest: {result.source_path}")
        fragment_path = Path(result.manifest_path)
        fragment = pq.read_table(fragment_path)
        if manifest_schema is None:
            manifest_schema = fragment.schema
        elif not fragment.schema.equals(manifest_schema, check_metadata=True):
            raise ValueError(
                f"Manifest schema mismatch for worker output {fragment_path}"
            )

        rows = fragment.to_pylist()
        shard_mapping: dict[str, str] = {}
        for relative_path in dict.fromkeys(str(row["shard_path"]) for row in rows):
            source_shard = Path(result.job_dir) / relative_path
            if not source_shard.is_file():
                raise FileNotFoundError(source_shard)
            final_name = f"shard-{shard_index:06d}.arrow"
            final_relative = f"shards/{final_name}"
            shard_mapping[relative_path] = final_relative
            shard_plan.append((source_shard, final_relative))
            shard_index += 1

        for row in rows:
            sample_id = str(row["sample_id"])
            source_uri = str(row["source_uri"])
            previous = sample_sources.get(sample_id)
            if previous is not None:
                raise ValueError(
                    "Duplicate sample ID encountered while merging worker outputs: "
                    f"{sample_id!r}. First source: {previous!r}; "
                    f"duplicate source: {source_uri!r}"
                )
            sample_sources[sample_id] = source_uri
            row["shard_path"] = shard_mapping[str(row["shard_path"])]
            merged_rows.append(row)

    if manifest_schema is None or not merged_rows:
        raise ValueError("No EEG examples were available for final merge")

    manifest = pa.Table.from_pylist(merged_rows, schema=manifest_schema)
    manifest_path = output_dir / "manifest.parquet"
    manifest_tmp = _temporary_path(manifest_path)
    manifest_tmp.unlink(missing_ok=True)
    pq.write_table(
        manifest,
        manifest_tmp,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    split_examples: dict[str, int] = {}
    for row in merged_rows:
        split = str(row["split"] or "unspecified")
        split_examples[split] = split_examples.get(split, 0) + 1

    try:
        _write_publication_plan(shard_plan, output_dir=output_dir)
        _publish_shards(
            shard_plan,
            output_dir=output_dir,
            show_progress=show_progress,
        )

        if progress_enabled:
            print(
                f"Finalizing manifest and summary for {len(merged_rows):,} examples...",
                flush=True,
            )

        total_signal_bytes = sum(result.total_signal_bytes for result in successful)
        wall_seconds = time.perf_counter() - started_at
        timing = {
            "wall_seconds": wall_seconds,
            "processing_seconds": processing_seconds,
            "merge_seconds": max(0.0, wall_seconds - processing_seconds),
            "recordings_per_second": processed_recordings / max(wall_seconds, 1e-12),
            "examples_per_second": len(merged_rows) / max(wall_seconds, 1e-12),
            "signal_mib_per_second": (
                total_signal_bytes / (1024**2) / max(wall_seconds, 1e-12)
            ),
        }
        summary = HarmonizationSummary(
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            examples=len(merged_rows),
            shards=shard_index,
            datasets=sorted({str(row["dataset_id"]) for row in merged_rows}),
            source_formats=sorted({str(row["source_format"]) for row in merged_rows}),
            split_examples=split_examples,
            total_signal_bytes=total_signal_bytes,
            source_audit=audit,
            timing=timing,
        )

        summary_path = output_dir / "summary.json"
        summary_tmp = _temporary_path(summary_path)
        summary_tmp.unlink(missing_ok=True)
        summary_tmp.write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        manifest_tmp.replace(manifest_path)
        summary_tmp.replace(summary_path)
        _publication_plan_path(output_dir).unlink(missing_ok=True)
    except Exception:
        manifest_tmp.unlink(missing_ok=True)
        (output_dir / "manifest.parquet").unlink(missing_ok=True)
        _temporary_path(output_dir / "summary.json").unlink(missing_ok=True)
        (output_dir / "summary.json").unlink(missing_ok=True)
        _rollback_published_shards(shard_plan, output_dir=output_dir)
        _publication_plan_path(output_dir).unlink(missing_ok=True)
        raise

    shutil.rmtree(output_dir / "_work", ignore_errors=True)
    return summary


def harmonize_recordings(
    *,
    source_kind: SourceKind,
    source_paths: Sequence[str | Path],
    dataset_root: str | Path,
    output_dir: str | Path,
    reader_options: dict[str, Any],
    records_per_batch: int,
    batches_per_shard: int,
    num_workers: int = 1,
    target_job_bytes: int = 0,
    max_recordings_per_job: int = 128,
    overwrite: bool = False,
    resume: bool = False,
    skip_invalid_recordings: bool = False,
    show_progress: bool = True,
) -> HarmonizationSummary:
    """Shared recording-parallel harmonization engine.

    Recordings are deterministically bundled by estimated source size. Each
    worker keeps one Arrow writer open across its bundle, allowing shards to
    contain many recordings. Only rank 0 merges outputs and renders progress.
    """

    if records_per_batch <= 0 or batches_per_shard <= 0:
        raise ValueError("records_per_batch and batches_per_shard must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    output_path = Path(output_dir)
    work_dir = _prepare_output(output_path, overwrite=overwrite, resume=resume)
    ordered_paths = [Path(path) for path in source_paths]
    if not ordered_paths:
        raise FileNotFoundError(f"No recordings were discovered for {source_kind}")

    bundles = plan_recording_bundles(
        ordered_paths,
        target_job_bytes=target_job_bytes,
        max_recordings_per_job=max_recordings_per_job,
    )
    jobs = [
        HarmonizationJob(
            index=index,
            source_kind=source_kind,
            source_paths=tuple(str(path) for path in bundle),
            dataset_root=str(dataset_root),
            work_dir=str(work_dir / f"job-{index:06d}"),
            reader_options=dict(reader_options),
            records_per_batch=records_per_batch,
            batches_per_shard=batches_per_shard,
        )
        for index, bundle in enumerate(bundles)
    ]

    started = time.perf_counter()
    results = _collect_results(
        jobs,
        num_workers=num_workers,
        show_progress=show_progress,
        description=f"Harmonizing {source_kind}",
    )
    processing_seconds = time.perf_counter() - started
    return _merge_job_outputs(
        results,
        output_dir=output_path,
        source_kind=source_kind,
        num_workers=num_workers,
        processing_seconds=processing_seconds,
        started_at=started,
        skip_invalid_recordings=skip_invalid_recordings,
        show_progress=show_progress,
    )
