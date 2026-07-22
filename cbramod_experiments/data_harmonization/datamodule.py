from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..datasets.shumi import SHUH5Dataset
from ..utils.utils import seed_worker
from .storage import (
    ArrowEEGDataset,
    StreamingArrowEEGDataset,
)


DataBackend = Literal[
    "hdf5",
    "arrow",
    "arrow_streaming",
]


class EEGDataModule:
    def __init__(
        self,
        path: str | Path,
        *,
        backend: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
        streaming_shuffle_buffer_size: int = 2048,
        prefetch_factor: int = 2,
        require_labels: bool = True,
    ) -> None:
        self.path = Path(path)
        self.backend = backend
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.streaming_shuffle_buffer_size = streaming_shuffle_buffer_size
        self.prefetch_factor = prefetch_factor
        self.require_labels = require_labels

    def _dataset(
        self,
        split: str | None,
        *,
        shuffle: bool,
    ) -> Dataset | IterableDataset:
        if self.backend == "hdf5":
            if split is None:
                raise ValueError("The HDF5 backend requires train, val, or test.")
            return SHUH5Dataset(self.path, split)

        if self.backend == "arrow":
            return ArrowEEGDataset(
                self.path,
                split=split,
                require_labels=self.require_labels,
            )

        if self.backend == "arrow_streaming":
            return StreamingArrowEEGDataset(
                self.path,
                split=split,
                require_labels=self.require_labels,
                shuffle_shards=shuffle,
                shuffle_buffer_size=(
                    self.streaming_shuffle_buffer_size if shuffle else 0
                ),
                seed=self.seed,
            )

        raise ValueError(f"Unsupported backend: {self.backend!r}")

    def loader(
        self,
        split: str | None,
        *,
        shuffle: bool | None = None,
    ) -> DataLoader:
        if split not in {None, "train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        if shuffle is None:
            shuffle = split == "train"

        dataset = self._dataset(split, shuffle=shuffle)
        is_streaming = isinstance(dataset, IterableDataset)

        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "worker_init_fn": seed_worker,
        }

        if not is_streaming:
            kwargs["shuffle"] = shuffle

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**kwargs)

    def loaders(self) -> dict[str, DataLoader]:
        return {split: self.loader(split) for split in ("train", "val", "test")}
