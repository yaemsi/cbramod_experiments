from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
from scipy import signal
import torch
from torch.utils.data import DataLoader, Dataset

from cbramod_experiments.utils import seed_worker


@dataclass(frozen=True)
class PreprocessingSummary:
    source_files: int
    examples: int
    channels: int
    original_points: int
    target_points: int
    output_path: str


_EXPLICIT_SUBJECT_PATTERN = re.compile(
    r"(?i)sub(?:ject)?[-_ ]*0*(?P<subject>[1-9]|1\d|2[0-5])(?:\D|$)"
)
_FALLBACK_SUBJECT_PATTERN = re.compile(
    r"0*(?P<subject>[1-9]|1\d|2[0-5])(?:\D|$)"
)
_SESSION_PATTERN = re.compile(r"(?i)(?:ses|sess|session)[-_ ]*0*(?P<session>[1-9]\d*)")


def subject_split(subject_id: int) -> str:
    if 1 <= subject_id <= 15:
        return "train"
    if 16 <= subject_id <= 20:
        return "val"
    if 21 <= subject_id <= 25:
        return "test"
    raise ValueError(f"SHU-MI subject must be in [1, 25], received {subject_id}")


def parse_subject_id(filename: str) -> int:
    stem = Path(filename).stem
    match = _EXPLICIT_SUBJECT_PATTERN.search(stem)
    if match is None:
        match = _FALLBACK_SUBJECT_PATTERN.search(stem)
    if match is None:
        raise ValueError(
            f"Could not parse a subject ID from {filename!r}. "
            "Pass a dataset-specific subject regex after inspecting the extracted filenames."
        )
    return int(match.group("subject"))


def parse_session_id(filename: str) -> int:
    match = _SESSION_PATTERN.search(Path(filename).stem)
    if match is not None:
        return int(match.group("session"))
    numbers = [int(value) for value in re.findall(r"\d+", Path(filename).stem)]
    return numbers[1] if len(numbers) >= 2 else -1


def preprocess_shu(
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    target_sampling_rate: int = 200,
    original_sampling_rate: int = 250,
    window_seconds: float = 4.0,
    amplitude_scale: float = 100.0,
    overwrite: bool = False,
) -> PreprocessingSummary:
    """Convert SHU-MI MATLAB files into one versioned, chunked HDF5 dataset."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    files = sorted(raw_dir.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found below {raw_dir}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use overwrite=True")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    target_points = int(round(target_sampling_rate * window_seconds))
    expected_original_points = int(round(original_sampling_rate * window_seconds))
    total_examples = 0
    channels: int | None = None
    observed_original_points: int | None = None
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        signals = None
        labels = handle.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="i1")
        subjects = handle.create_dataset("subject_ids", shape=(0,), maxshape=(None,), dtype="i2")
        sessions = handle.create_dataset("session_ids", shape=(0,), maxshape=(None,), dtype="i2")
        trial_ids = handle.create_dataset("trial_ids", shape=(0,), maxshape=(None,), dtype="i4")
        source_files = handle.create_dataset(
            "source_files", shape=(0,), maxshape=(None,), dtype=string_dtype
        )

        for source_path in files:
            payload = scipy.io.loadmat(source_path)
            if "data" not in payload or "labels" not in payload:
                raise KeyError(f"{source_path} must contain MATLAB variables 'data' and 'labels'")
            eeg = np.asarray(payload["data"])
            target = np.asarray(payload["labels"]).reshape(-1)
            if eeg.ndim != 3:
                raise ValueError(f"Expected [trials, channels, time] in {source_path}, got {eeg.shape}")
            if eeg.shape[0] != target.shape[0]:
                raise ValueError(f"Trial/label mismatch in {source_path}: {eeg.shape[0]} vs {target.shape[0]}")
            if channels is None:
                channels = int(eeg.shape[1])
                observed_original_points = int(eeg.shape[2])
                signals = handle.create_dataset(
                    "signals",
                    shape=(0, channels, target_points),
                    maxshape=(None, channels, target_points),
                    chunks=(1, channels, target_points),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                    dtype="f4",
                )
            if eeg.shape[1] != channels:
                raise ValueError(f"Channel count changed in {source_path}: {eeg.shape[1]} vs {channels}")
            if eeg.shape[2] != observed_original_points:
                raise ValueError(
                    f"Signal length changed in {source_path}: {eeg.shape[2]} vs {observed_original_points}"
                )
            if eeg.shape[2] != expected_original_points:
                raise ValueError(
                    f"Expected {expected_original_points} samples at {original_sampling_rate} Hz for "
                    f"{window_seconds}s, but {source_path} contains {eeg.shape[2]}"
                )

            subject_id = parse_subject_id(source_path.name)
            session_id = parse_session_id(source_path.name)
            split = subject_split(subject_id)
            eeg = signal.resample(eeg, target_points, axis=-1).astype(np.float32, copy=False)
            target = target.astype(np.int64, copy=False)
            if target.min() == 1 and target.max() == 2:
                target = target - 1
            if not np.isin(target, [0, 1]).all():
                raise ValueError(f"Expected binary labels encoded as 0/1 or 1/2 in {source_path}")

            start = total_examples
            end = start + eeg.shape[0]
            assert signals is not None
            for dataset in (signals, labels, subjects, sessions, trial_ids, source_files):
                dataset.resize((end,) + dataset.shape[1:])
            signals[start:end] = eeg
            labels[start:end] = target.astype(np.int8)
            subjects[start:end] = subject_id
            sessions[start:end] = session_id
            trial_ids[start:end] = np.arange(eeg.shape[0], dtype=np.int32)
            source_files[start:end] = source_path.name
            split_indices[split].extend(range(start, end))
            total_examples = end

        split_group = handle.create_group("splits")
        for split, indices in split_indices.items():
            split_group.create_dataset(split, data=np.asarray(indices, dtype=np.int64))
        handle.attrs.update(
            {
                "dataset": "SHU-MI",
                "schema_version": "1.0",
                "original_sampling_rate": original_sampling_rate,
                "target_sampling_rate": target_sampling_rate,
                "window_seconds": window_seconds,
                "amplitude_scale": amplitude_scale,
                "split_protocol": "subjects 1-15 train, 16-20 val, 21-25 test",
                "source_file_count": len(files),
            }
        )

    summary = PreprocessingSummary(
        source_files=len(files),
        examples=total_examples,
        channels=int(channels or 0),
        original_points=int(observed_original_points or 0),
        target_points=target_points,
        output_path=str(output_path),
    )
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary


class SHUH5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Worker-safe lazy reader for the processed SHU-MI HDF5 file."""

    def __init__(self, path: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.path = str(path)
        self.split = split
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.indices = np.asarray(handle[f"splits/{split}"], dtype=np.int64)
            self.amplitude_scale = float(handle.attrs.get("amplitude_scale", 1.0))

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        handle = self._file()
        eeg = np.asarray(handle["signals"][row], dtype=np.float32) / self.amplitude_scale
        label = int(handle["labels"][row])
        return torch.from_numpy(eeg), torch.tensor(label, dtype=torch.long)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()


class SHUDataModule:
    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
    ) -> None:
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def loaders(self) -> dict[str, DataLoader]:
        generator = torch.Generator().manual_seed(self.seed)
        loaders: dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            loaders[split] = DataLoader(
                SHUH5Dataset(self.path, split),
                batch_size=self.batch_size,
                shuffle=split == "train",
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=seed_worker,
                generator=generator if split == "train" else None,
                drop_last=False,
            )
        return loaders
