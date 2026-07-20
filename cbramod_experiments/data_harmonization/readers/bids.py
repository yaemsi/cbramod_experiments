from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import mne
import numpy as np
import pandas as pd

from ..schema import DEFAULT_PREPROCESSING_VERSION, EEGEvent, EEGRecording, EEGWindow
from ..transforms import (
    normalize_channel_names,
    resample_recording,
    select_channels,
    sliding_windows,
)

_ENTITY_PATTERN = re.compile(r"(?:^|_)(sub|ses|task)[-_]([^_]+)")
_SUPPORTED_EXTENSIONS = {".edf", ".bdf", ".set"}


def parse_bids_entities(path: str | Path) -> dict[str, str]:
    return {key: value for key, value in _ENTITY_PATTERN.findall(Path(path).name)}


def _read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG format: {path.suffix}")


def _exact_sidecar(signal_path: Path, suffix: str) -> Path:
    stem = signal_path.name
    for extension in _SUPPORTED_EXTENSIONS:
        ending = f"_eeg{extension}"
        if stem.lower().endswith(ending):
            return signal_path.with_name(stem[: -len(ending)] + suffix)
    return signal_path.with_suffix(suffix)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class BIDSReader:
    """Read BIDS-like EDF/BDF/SET recordings into the canonical schema.

    This intentionally avoids requiring ``mne-bids`` for the POC. It supports
    exact recording sidecars and common root-level task JSON inheritance.
    Production code should use a full BIDS resolver/validator.
    """

    def __init__(self, dataset_id: str = "hbn") -> None:
        self.dataset_id = dataset_id

    def discover(
        self,
        root: str | Path,
        *,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        limit_recordings: int | None = None,
    ) -> list[Path]:
        root_path = Path(root)
        files = sorted(
            path
            for path in root_path.rglob("*_eeg.*")
            if path.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        subject_filter = {item.removeprefix("sub-") for item in subjects or ()}
        task_filter = {item.removeprefix("task-") for item in tasks or ()}
        selected: list[Path] = []
        for path in files:
            entities = parse_bids_entities(path)
            if subject_filter and entities.get("sub") not in subject_filter:
                continue
            if task_filter and entities.get("task") not in task_filter:
                continue
            selected.append(path)
            if limit_recordings is not None and len(selected) >= limit_recordings:
                break
        return selected

    def read_recording(self, path: str | Path, *, root: str | Path) -> EEGRecording:
        signal_path = Path(path)
        root_path = Path(root)
        entities = parse_bids_entities(signal_path)
        if "sub" not in entities:
            raise ValueError(f"Cannot parse BIDS subject from {signal_path}")
        raw = _read_raw(signal_path)
        channels_path = _exact_sidecar(signal_path, "_channels.tsv")
        channel_metadata: list[dict[str, object]] = []
        if channels_path.exists():
            channel_table = pd.read_csv(channels_path, sep="\t")
            channel_metadata = channel_table.to_dict(orient="records")
            type_mapping: dict[str, str] = {}
            supported_types = {
                "EEG": "eeg",
                "EOG": "eog",
                "ECG": "ecg",
                "EMG": "emg",
                "TRIG": "stim",
                "STIM": "stim",
                "MISC": "misc",
            }
            for row in channel_metadata:
                name = row.get("name")
                channel_type = row.get("type")
                if isinstance(name, str) and isinstance(channel_type, str):
                    mapped = supported_types.get(channel_type.upper())
                    if mapped is not None and name in raw.ch_names:
                        type_mapping[name] = mapped
            if type_mapping:
                raw.set_channel_types(type_mapping, on_unit_change="ignore")

        eeg_indices = mne.pick_types(
            raw.info,
            eeg=True,
            meg=False,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude="bads",
        )
        if len(eeg_indices) == 0:
            raise ValueError(f"No EEG channels found in {signal_path}")
        data_uv = cast(np.ndarray, raw.get_data(picks=eeg_indices)) * 1e6
        eeg_channel_names = tuple(raw.ch_names[int(index)] for index in eeg_indices)

        events_path = _exact_sidecar(signal_path, "_events.tsv")
        events: list[EEGEvent] = []
        if events_path.exists():
            table = pd.read_csv(events_path, sep="\t")
            for row in cast(list[dict[str, Any]], table.to_dict(orient="records")):
                onset_value = row.get("onset", 0.0)
                duration_value = row.get("duration", 0.0)
                onset = float(0.0 if onset_value is None else onset_value)
                duration = float(0.0 if duration_value is None else duration_value)
                event_type = str(row.get("trial_type", row.get("value", "event")))
                value = row.get("value")
                if value is not None and bool(pd.isna(value)):
                    value = None
                events.append(
                    EEGEvent(
                        onset_seconds=onset,
                        duration_seconds=max(0.0, duration),
                        event_type=event_type,
                        value=value,
                    )
                )

        recording_json = _load_json(_exact_sidecar(signal_path, "_eeg.json"))
        task = entities.get("task")
        inherited_json = _load_json(root_path / f"task-{task}_eeg.json") if task else {}
        result = EEGRecording(
            signal=np.asarray(data_uv, dtype=np.float32),
            sampling_rate_hz=float(raw.info["sfreq"]),
            channel_names=eeg_channel_names,
            dataset_id=self.dataset_id,
            subject_id=f"sub-{entities['sub']}",
            session_id=(f"ses-{entities['ses']}" if "ses" in entities else None),
            task=(f"task-{task}" if task else None),
            events=tuple(events),
            source_uri=str(signal_path),
            source_format=signal_path.suffix.lower().lstrip("."),
            metadata={
                "recording_json": {**inherited_json, **recording_json},
                "channels": channel_metadata,
                "events_path": str(events_path) if events_path.exists() else None,
            },
        )
        result = normalize_channel_names(result)
        result.validate()
        return result

    def iter_windows(
        self,
        root: str | Path,
        *,
        target_sampling_rate_hz: float | None,
        window_seconds: float,
        stride_seconds: float,
        channel_policy: str = "preserve",
        channels: Sequence[str] | None = None,
        allow_missing_channels: bool = False,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        limit_recordings: int | None = None,
        amplitude_scale: float = 100.0,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
    ) -> Iterable[EEGWindow]:
        files = self.discover(
            root,
            subjects=subjects,
            tasks=tasks,
            limit_recordings=limit_recordings,
        )
        if not files:
            raise FileNotFoundError(f"No BIDS EEG recordings found below {root}")

        for path in files:
            recording = self.read_recording(path, root=root)
            if target_sampling_rate_hz is not None:
                recording = resample_recording(recording, target_sampling_rate_hz)
            mask = np.ones(len(recording.channel_names), dtype=np.bool_)
            if channel_policy == "select":
                if not channels:
                    raise ValueError(
                        "channels must be provided for channel_policy='select'"
                    )
                recording, mask = select_channels(
                    recording,
                    channels,
                    allow_missing=allow_missing_channels,
                )
            elif channel_policy != "preserve":
                raise ValueError("channel_policy must be either 'preserve' or 'select'")
            yield from sliding_windows(
                recording,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                split=None,
                amplitude_scale=amplitude_scale,
                preprocessing_version=preprocessing_version,
                channel_mask=mask,
            )
