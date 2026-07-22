from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..datasets.shumi import SHUH5Dataset
from ..utils.utils import seed_worker
from .storage import (
    ArrowBlockShuffleSampler,
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
        backend: DataBackend,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
        streaming_shuffle_buffer_size: int = 2048,
    ) -> None:
        if backend not in {
            "hdf5",
            "arrow",
            "arrow_streaming",
        }:
            raise ValueError(f"Unsupported data backend: {backend}")

        if streaming_shuffle_buffer_size <= 0:
            raise ValueError("streaming_shuffle_buffer_size must be positive")

        self.path = Path(path)
        self.backend = backend
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.streaming_shuffle_buffer_size = streaming_shuffle_buffer_size

    def _dataset(self, split: str) -> Dataset | IterableDataset:
        if self.backend == "hdf5":
            return SHUH5Dataset(self.path, split)

        if self.backend == "arrow":
            return ArrowEEGDataset(self.path, split)

        if self.backend == "arrow_streaming":
            return StreamingArrowEEGDataset(
                self.path,
                split=split,
                require_labels=True,
                shuffle_shards=split == "train",
                shuffle_buffer_size=(
                    self.streaming_shuffle_buffer_size if split == "train" else 0
                ),
                seed=self.seed,
            )

        raise ValueError(f"Unsupported backend: {self.backend!r}")

    def loaders(self) -> dict[str, DataLoader]:
        generator = torch.Generator().manual_seed(self.seed)
        loaders: dict[str, DataLoader] = {}

        for split in ("train", "val", "test"):
            dataset = self._dataset(split)

            is_streaming = isinstance(
                dataset,
                IterableDataset,
            )

            sampler = None
            shuffle = split == "train" and not is_streaming

            if split == "train" and isinstance(dataset, ArrowEEGDataset):
                sampler = ArrowBlockShuffleSampler(
                    dataset,
                    seed=self.seed,
                )
                shuffle = False

            loader_kwargs = {
                "dataset": dataset,
                "batch_size": self.batch_size,
                "shuffle": shuffle,
                "sampler": sampler,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "worker_init_fn": seed_worker,
                "drop_last": False,
            }

            # Iterable datasets manage ordering internally.
            if not is_streaming and split == "train":
                loader_kwargs["generator"] = generator

            if self.num_workers > 0:
                loader_kwargs["prefetch_factor"] = 2

            loaders[split] = DataLoader(**loader_kwargs)

        return loaders
