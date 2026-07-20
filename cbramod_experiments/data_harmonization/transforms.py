from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
from scipy import signal as scipy_signal

from .schema import EEGRecording, EEGWindow


def normalize_channel_name(name: str) -> str:
    """Normalize harmless spelling differences without changing montage identity."""
    return name.strip().replace(" ", "").replace("EEG", "").strip("-_")


def normalize_channel_names(recording: EEGRecording) -> EEGRecording:
    names = tuple(normalize_channel_name(name) for name in recording.channel_names)
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("Channel-name normalization produced duplicate channels")
    result = replace(recording, channel_names=names)
    result.validate()
    return result


def resample_signal(
    signal: np.ndarray,
    *,
    original_sampling_rate_hz: float,
    target_sampling_rate_hz: float,
) -> np.ndarray:
    if target_sampling_rate_hz <= 0 or original_sampling_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive")
    if np.isclose(original_sampling_rate_hz, target_sampling_rate_hz):
        return np.asarray(signal, dtype=np.float32)
    target_points = int(
        round(signal.shape[-1] * target_sampling_rate_hz / original_sampling_rate_hz)
    )
    return np.asarray(
        scipy_signal.resample(signal, target_points, axis=-1), dtype=np.float32
    )


def resample_recording(
    recording: EEGRecording, target_sampling_rate_hz: float
) -> EEGRecording:
    output = resample_signal(
        recording.signal,
        original_sampling_rate_hz=recording.sampling_rate_hz,
        target_sampling_rate_hz=target_sampling_rate_hz,
    )
    result = replace(
        recording,
        signal=output,
        sampling_rate_hz=float(target_sampling_rate_hz),
    )
    result.validate()
    return result


def select_channels(
    recording: EEGRecording,
    requested_channels: Sequence[str],
    *,
    allow_missing: bool = False,
) -> tuple[EEGRecording, np.ndarray]:
    """Select/reorder channels and return a mask indicating observed channels.

    When ``allow_missing`` is true, missing channels are zero-filled. This is a
    useful POC policy, but interpolation should be considered for production.
    """
    normalized_requested = tuple(normalize_channel_name(c) for c in requested_channels)
    lookup = {
        normalize_channel_name(name).casefold(): index
        for index, name in enumerate(recording.channel_names)
    }
    rows: list[np.ndarray] = []
    mask: list[bool] = []
    for name in normalized_requested:
        index = lookup.get(name.casefold())
        if index is None:
            if not allow_missing:
                raise KeyError(
                    f"Requested channel {name!r} is absent from {recording.source_uri}"
                )
            rows.append(np.zeros(recording.signal.shape[1], dtype=np.float32))
            mask.append(False)
        else:
            rows.append(np.asarray(recording.signal[index], dtype=np.float32))
            mask.append(True)
    output = np.stack(rows, axis=0)
    result = replace(
        recording,
        signal=output,
        channel_names=normalized_requested,
    )
    result.validate()
    return result, np.asarray(mask, dtype=np.bool_)


def simple_quality_flags(signal: np.ndarray) -> tuple[str, ...]:
    flags: list[str] = []
    if not np.isfinite(signal).all():
        flags.append("non_finite")
    per_channel_std = np.std(signal, axis=-1)
    if np.any(per_channel_std < 1e-6):
        flags.append("flat_channel")
    if np.max(np.abs(signal), initial=0.0) > 2_000.0:
        flags.append("extreme_amplitude")
    return tuple(flags)


def sliding_windows(
    recording: EEGRecording,
    *,
    window_seconds: float,
    stride_seconds: float,
    split: str | None,
    amplitude_scale: float,
    preprocessing_version: str,
    channel_mask: np.ndarray | None = None,
) -> Iterable[EEGWindow]:
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Window and stride durations must be positive")
    window_points = int(round(window_seconds * recording.sampling_rate_hz))
    stride_points = int(round(stride_seconds * recording.sampling_rate_hz))
    if window_points <= 0 or stride_points <= 0:
        raise ValueError("Window and stride resolve to zero samples")
    if channel_mask is None:
        channel_mask = np.ones(len(recording.channel_names), dtype=np.bool_)

    total_points = recording.signal.shape[1]
    for start in range(0, max(0, total_points - window_points + 1), stride_points):
        stop = start + window_points
        sample_id = (
            f"{recording.dataset_id}:{recording.subject_id}:"
            f"{recording.session_id or 'none'}:{recording.task or 'none'}:"
            f"sample-{start:09d}"
        )
        window = np.asarray(recording.signal[:, start:stop], dtype=np.float32)
        result = EEGWindow(
            signal=window,
            sampling_rate_hz=recording.sampling_rate_hz,
            channel_names=recording.channel_names,
            channel_mask=np.asarray(channel_mask, dtype=np.bool_),
            dataset_id=recording.dataset_id,
            subject_id=recording.subject_id,
            session_id=recording.session_id,
            task=recording.task,
            start_seconds=start / recording.sampling_rate_hz,
            duration_seconds=window_seconds,
            label=None,
            split=split,
            source_uri=recording.source_uri,
            source_format=recording.source_format,
            sample_id=sample_id,
            preprocessing_version=preprocessing_version,
            amplitude_scale=amplitude_scale,
            quality_flags=simple_quality_flags(window),
            metadata=recording.metadata,
        )
        result.validate()
        yield result
