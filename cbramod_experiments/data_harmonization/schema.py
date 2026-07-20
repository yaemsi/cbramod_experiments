from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CANONICAL_SCHEMA_VERSION = "1.0"
DEFAULT_PREPROCESSING_VERSION = "harmonized-eeg-v1"


@dataclass(frozen=True)
class EEGEvent:
    """One event attached to a continuous EEG recording."""

    onset_seconds: float
    duration_seconds: float
    event_type: str
    value: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EEGRecording:
    """Source-independent representation of one continuous EEG recording.

    Signals are stored in microvolts and shaped ``[channels, time]``.
    """

    signal: np.ndarray
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    dataset_id: str
    subject_id: str
    session_id: str | None
    task: str | None
    events: tuple[EEGEvent, ...] = ()
    source_uri: str = ""
    source_format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError(
                f"EEGRecording.signal must be [channels, time], got {self.signal.shape}"
            )
        if self.signal.shape[0] != len(self.channel_names):
            raise ValueError(
                "Number of channel names does not match the signal channel dimension"
            )
        if self.signal.shape[1] <= 0:
            raise ValueError("EEGRecording must contain at least one time sample")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if not np.isfinite(self.signal).all():
            raise ValueError("EEGRecording contains NaN or infinite values")
        if not self.subject_id:
            raise ValueError("subject_id cannot be empty")


@dataclass(frozen=True)
class EEGWindow:
    """Canonical model-ready EEG example.

    The signal remains in microvolts. A storage backend may apply an additional
    amplitude scale at read time, but the scale is recorded explicitly.
    """

    signal: np.ndarray
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    channel_mask: np.ndarray
    dataset_id: str
    subject_id: str
    session_id: str | None
    task: str | None
    start_seconds: float
    duration_seconds: float
    label: int | None
    split: str | None
    source_uri: str
    source_format: str
    sample_id: str
    preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION
    amplitude_scale: float = 1.0
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError(
                f"EEGWindow.signal must be [channels, time], got {self.signal.shape}"
            )
        if self.signal.shape[0] != len(self.channel_names):
            raise ValueError(
                "Number of channel names does not match the signal channel dimension"
            )
        if self.channel_mask.shape != (self.signal.shape[0],):
            raise ValueError(
                "channel_mask must have one entry per signal channel, got "
                f"{self.channel_mask.shape} for {self.signal.shape[0]} channels"
            )
        if self.signal.shape[1] <= 0:
            raise ValueError("EEGWindow must contain at least one time sample")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.amplitude_scale <= 0:
            raise ValueError("amplitude_scale must be positive")
        if not np.isfinite(self.signal).all():
            raise ValueError("EEGWindow contains NaN or infinite values")
        if self.split not in {None, "train", "val", "test"}:
            raise ValueError(f"Unknown split: {self.split}")
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
