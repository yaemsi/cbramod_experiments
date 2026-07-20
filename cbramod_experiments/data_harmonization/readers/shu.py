from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, cast

import mne
import numpy as np
import pandas as pd
import scipy.io

from ...datasets.shumi import parse_session_id, parse_subject_id, subject_split
from ..schema import DEFAULT_PREPROCESSING_VERSION, EEGWindow
from ..transforms import resample_signal, simple_quality_flags


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


class SHUMatReader:
    """Read classification-ready SHU-MI MATLAB trials."""

    dataset_id = "shu-mi"

    def discover(self, root: str | Path) -> list[Path]:
        files = sorted(Path(root).rglob("*_eeg.mat"))
        if not files:
            files = sorted(Path(root).rglob("*.mat"))
        return files

    def iter_windows(
        self,
        root: str | Path,
        *,
        target_sampling_rate_hz: float = 200.0,
        original_sampling_rate_hz: float = 250.0,
        amplitude_scale: float = 100.0,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    ) -> Iterable[EEGWindow]:
        files = self.discover(root)
        if not files:
            raise FileNotFoundError(f"No SHU-MI MATLAB files found below {root}")

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
                    f"Expected {len(SHU_CHANNEL_NAMES)} SHU channels, got {signals.shape[1]}"
                )
            if labels.size and labels.min() == 1 and labels.max() == 2:
                labels = labels - 1
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"Unexpected labels in {path}: {np.unique(labels)}")

            resampled = resample_signal(
                signals,
                original_sampling_rate_hz=original_sampling_rate_hz,
                target_sampling_rate_hz=target_sampling_rate_hz,
            )
            subject_number = parse_subject_id(path.name)
            session_number = parse_session_id(path.name)
            split = subject_split(subject_number)
            duration_seconds = resampled.shape[-1] / target_sampling_rate_hz
            for trial_index, (trial, label) in enumerate(
                zip(resampled, labels, strict=True)
            ):
                result = EEGWindow(
                    signal=np.asarray(trial, dtype=np.float32),
                    sampling_rate_hz=float(target_sampling_rate_hz),
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
    """Reconstruct SHU-MI trials from continuous EDF plus event TSV files."""

    dataset_id = "shu-mi"

    def discover(self, root: str | Path) -> list[Path]:
        return sorted(Path(root).rglob("*_eeg.edf"))

    def iter_windows(
        self,
        root: str | Path,
        *,
        events_root: str | Path | None = None,
        target_sampling_rate_hz: float = 200.0,
        amplitude_scale: float = 100.0,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    ) -> Iterable[EEGWindow]:
        files = self.discover(root)
        if not files:
            raise FileNotFoundError(f"No SHU-MI EDF files found below {root}")
        event_root_path = Path(events_root) if events_root else Path(root)

        for path in files:
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

            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            source_rate = float(raw.info["sfreq"])
            continuous = cast(np.ndarray, raw.get_data()) * 1e6
            continuous = np.asarray(continuous, dtype=np.float32)
            channel_names = tuple(raw.ch_names)
            events = pd.read_csv(event_path, sep="\t")
            required = {"duration", "value"}
            if not required.issubset(events.columns):
                raise ValueError(
                    f"{event_path} is missing columns {sorted(required - set(events.columns))}"
                )
            position_column = "sample" if "sample" in events.columns else "onset"
            subject_number = parse_subject_id(path.name)
            session_number = parse_session_id(path.name)
            split = subject_split(subject_number)

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
                    target_sampling_rate_hz=target_sampling_rate_hz,
                )
                label = int(row["value"])
                if label in {1, 2}:
                    label -= 1
                result = EEGWindow(
                    signal=trial,
                    sampling_rate_hz=float(target_sampling_rate_hz),
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
                yield result
