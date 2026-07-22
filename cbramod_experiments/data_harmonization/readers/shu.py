from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import mne
import numpy as np
import pandas as pd
import scipy.io

from ...datasets.shumi import parse_session_id, parse_subject_id, subject_split
from ..schema import DEFAULT_PREPROCESSING_VERSION, EEGWindow
from ..transforms import resample_signal, simple_quality_flags


LOGGER = logging.getLogger(__name__)

SHU_CHANNEL_NAMES: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "Cz",
    "C3",
    "C4",
    "T3",
    "T4",
    "A1",
    "A2",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "Pz",
    "P3",
    "P4",
    "T5",
    "T6",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
)


@dataclass(frozen=True)
class EDFReadFailure:
    """Description of one SHU-MI EDF recording that could not be processed."""

    path: str
    error_type: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return cast(dict[str, str], asdict(self))


def _resolve_sampling_rate(
    value: float | str,
    *,
    source_sampling_rate_hz: float,
) -> float:
    if isinstance(value, str):
        if value.casefold() != "auto":
            raise ValueError(
                "target_sampling_rate_hz must be a positive number or 'auto'"
            )
        return float(source_sampling_rate_hz)
    resolved = float(value)
    if resolved <= 0:
        raise ValueError("target_sampling_rate_hz must be positive")
    return resolved


class SHUMatReader:
    """Read classification-ready SHU-MI MATLAB trials."""

    dataset_id = "shu-mi"

    def discover(self, root: str | Path) -> list[Path]:
        root_path = Path(root)
        if root_path.is_file():
            if root_path.suffix.casefold() != ".mat":
                raise ValueError(f"Expected a MAT file, received: {root_path}")
            return [root_path]
        files = sorted(root_path.rglob("*_eeg.mat"))
        if not files:
            files = sorted(root_path.rglob("*.mat"))
        return files

    def iter_windows(
        self,
        root: str | Path,
        *,
        target_sampling_rate_hz: float | str = 200.0,
        original_sampling_rate_hz: float = 250.0,
        amplitude_scale: float = 100.0,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        window_duration_seconds: float | None = None,
        window_stride_seconds: float | None = None,
    ) -> Iterable[EEGWindow]:
        del window_duration_seconds, window_stride_seconds
        files = self.discover(root)
        if not files:
            raise FileNotFoundError(f"No SHU-MI MATLAB files found below {root}")

        target_rate = _resolve_sampling_rate(
            target_sampling_rate_hz,
            source_sampling_rate_hz=original_sampling_rate_hz,
        )
        for path in files:
            payload = scipy.io.loadmat(path)
            if "data" not in payload or "labels" not in payload:
                raise KeyError(f"{path} must contain 'data' and 'labels'")
            signals = np.asarray(payload["data"], dtype=np.float32)
            labels = np.asarray(payload["labels"]).reshape(-1).astype(np.int64)
            if signals.ndim != 3:
                raise ValueError(
                    f"Expected [trials, channels, time] in {path}, got {signals.shape}"
                )
            if signals.shape[0] != labels.shape[0]:
                raise ValueError(f"Trial/label mismatch in {path}")
            if signals.shape[1] != len(SHU_CHANNEL_NAMES):
                raise ValueError(
                    f"Expected {len(SHU_CHANNEL_NAMES)} SHU channels, "
                    f"got {signals.shape[1]}"
                )
            if labels.size and labels.min() == 1 and labels.max() == 2:
                labels = labels - 1
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"Unexpected labels in {path}: {np.unique(labels)}")

            resampled = resample_signal(
                signals,
                original_sampling_rate_hz=original_sampling_rate_hz,
                target_sampling_rate_hz=target_rate,
            )
            subject_number = parse_subject_id(path.name)
            session_number = parse_session_id(path.name)
            split = subject_split(subject_number)
            duration_seconds = resampled.shape[-1] / target_rate
            for trial_index, (trial, label) in enumerate(
                zip(resampled, labels, strict=True)
            ):
                result = EEGWindow(
                    signal=np.asarray(trial, dtype=np.float32),
                    sampling_rate_hz=target_rate,
                    channel_names=SHU_CHANNEL_NAMES,
                    channel_mask=np.ones(len(SHU_CHANNEL_NAMES), dtype=np.bool_),
                    dataset_id=self.dataset_id,
                    subject_id=f"sub-{subject_number:03d}",
                    session_id=(
                        f"ses-{session_number:02d}" if session_number >= 0 else None
                    ),
                    task="motorimagery",
                    start_seconds=trial_index * duration_seconds,
                    duration_seconds=duration_seconds,
                    label=int(label),
                    split=split,
                    source_uri=str(path),
                    source_format="mat",
                    sample_id=(
                        f"shu-mi:sub-{subject_number:03d}:"
                        f"ses-{session_number:02d}:trial-{trial_index:04d}"
                    ),
                    preprocessing_version=preprocessing_version,
                    amplitude_scale=amplitude_scale,
                    quality_flags=simple_quality_flags(trial),
                    metadata={"trial_index": trial_index},
                )
                result.validate()
                yield result


class SHUEdfReader:
    """Reconstruct SHU-MI trials from continuous EDF plus event TSV files.

    Strict mode is the default: any unreadable or invalid recording aborts the
    operation. With ``strict=False``, expected source-data errors are logged,
    recorded in :attr:`failures`, and the affected recording is skipped.
    """

    dataset_id = "shu-mi"
    _EXPECTED_RECORDING_ERRORS = (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        pd.errors.ParserError,
    )

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.failures: list[EDFReadFailure] = []
        self.discovered_recordings = 0
        self.processed_recordings = 0

    def discover(self, root: str | Path) -> list[Path]:
        root_path = Path(root)
        if root_path.is_file():
            if root_path.suffix.casefold() != ".edf":
                raise ValueError(f"Expected an EDF file, received: {root_path}")
            return [root_path]
        return sorted(root_path.rglob("*_eeg.edf"))

    def audit_report(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "discovered_recordings": self.discovered_recordings,
            "processed_recordings": self.processed_recordings,
            "skipped_recordings": len(self.failures),
            "failures": [failure.to_dict() for failure in self.failures],
        }

    def _record_failure(self, path: Path, exc: BaseException) -> None:
        failure = EDFReadFailure(
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        self.failures.append(failure)
        if self.strict:
            LOGGER.error(
                "Invalid SHU-MI EDF recording %s (%s): %s",
                path,
                failure.error_type,
                failure.error,
            )
        else:
            LOGGER.warning(
                "Skipping invalid SHU-MI EDF recording %s (%s): %s",
                path,
                failure.error_type,
                failure.error,
            )

    def _event_path(self, path: Path, event_root_path: Path) -> Path:
        event_name = path.name.replace("_eeg.edf", "_events.tsv")
        candidates = [
            path.with_name(event_name),
            event_root_path / event_name,
            event_root_path / "events" / event_name,
        ]
        event_path = next((item for item in candidates if item.exists()), None)
        if event_path is None:
            raise FileNotFoundError(
                f"No matching events TSV for {path}; tried {candidates}"
            )
        return event_path

    def _read_recording_windows(
        self,
        path: Path,
        *,
        event_root_path: Path,
        target_sampling_rate_hz: float | str,
        amplitude_scale: float,
        preprocessing_version: str,
    ) -> list[EEGWindow]:
        event_path = self._event_path(path, event_root_path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        try:
            source_rate = float(raw.info["sfreq"])
            target_rate = _resolve_sampling_rate(
                target_sampling_rate_hz,
                source_sampling_rate_hz=source_rate,
            )
            continuous = cast(np.ndarray, raw.get_data()) * 1e6
            continuous = np.asarray(continuous, dtype=np.float32)
            channel_names = tuple(raw.ch_names)
        finally:
            raw.close()

        events = pd.read_csv(event_path, sep="\t")
        required = {"duration", "value"}
        if not required.issubset(events.columns):
            raise ValueError(
                f"{event_path} is missing columns "
                f"{sorted(required - set(events.columns))}"
            )
        position_column = "sample" if "sample" in events.columns else "onset"
        subject_number = parse_subject_id(path.name)
        session_number = parse_session_id(path.name)
        split = subject_split(subject_number)

        windows: list[EEGWindow] = []
        event_rows = cast(list[dict[str, Any]], events.to_dict(orient="records"))
        for trial_index, row in enumerate(event_rows):
            # SHU's TSV stores 1-based sample positions and duration in samples,
            # despite the BIDS-like column names.
            start = int(round(float(row[position_column]))) - 1
            duration_points = int(round(float(row["duration"])))
            stop = start + duration_points
            trial = continuous[:, start:stop]
            if trial.shape[-1] != duration_points:
                raise ValueError(
                    f"Event {trial_index} exceeds the bounds of {path}: {trial.shape}"
                )
            trial = resample_signal(
                trial,
                original_sampling_rate_hz=source_rate,
                target_sampling_rate_hz=target_rate,
            )
            label = int(row["value"])
            if label in {1, 2}:
                label -= 1
            result = EEGWindow(
                signal=trial,
                sampling_rate_hz=target_rate,
                channel_names=channel_names,
                channel_mask=np.ones(len(channel_names), dtype=np.bool_),
                dataset_id=self.dataset_id,
                subject_id=f"sub-{subject_number:03d}",
                session_id=(
                    f"ses-{session_number:02d}" if session_number >= 0 else None
                ),
                task="motorimagery",
                start_seconds=start / source_rate,
                duration_seconds=duration_points / source_rate,
                label=label,
                split=split,
                source_uri=str(path),
                source_format="edf",
                sample_id=(
                    f"shu-mi:sub-{subject_number:03d}:"
                    f"ses-{session_number:02d}:trial-{trial_index:04d}"
                ),
                preprocessing_version=preprocessing_version,
                amplitude_scale=amplitude_scale,
                quality_flags=simple_quality_flags(trial),
                metadata={
                    "trial_index": int(trial_index),
                    "event_file": str(event_path),
                    "trial_type": row.get("trial_type"),
                },
            )
            result.validate()
            windows.append(result)
        return windows

    def iter_windows(
        self,
        root: str | Path,
        *,
        events_root: str | Path | None = None,
        target_sampling_rate_hz: float | str = 200.0,
        amplitude_scale: float = 100.0,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        window_duration_seconds: float | None = None,
        window_stride_seconds: float | None = None,
    ) -> Iterable[EEGWindow]:
        del window_duration_seconds, window_stride_seconds
        files = self.discover(root)
        if not files:
            raise FileNotFoundError(f"No SHU-MI EDF files found below {root}")
        event_root_path = Path(events_root) if events_root else Path(root)
        self.discovered_recordings += len(files)

        for path in files:
            try:
                windows = self._read_recording_windows(
                    path,
                    event_root_path=event_root_path,
                    target_sampling_rate_hz=target_sampling_rate_hz,
                    amplitude_scale=amplitude_scale,
                    preprocessing_version=preprocessing_version,
                )
            except self._EXPECTED_RECORDING_ERRORS as exc:
                self._record_failure(path, exc)
                if self.strict:
                    raise
                continue

            self.processed_recordings += 1
            yield from windows
