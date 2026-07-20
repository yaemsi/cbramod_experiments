from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from ..datasets.shumi import SHUH5Dataset
from ..utils.utils import seed_worker
from .storage import ArrowBlockShuffleSampler, ArrowEEGDataset


DataBackend = Literal["hdf5", "arrow"]


class EEGDataModule:
    """Build identical PyTorch loaders from HDF5 or Arrow-backed samples."""

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
    ) -> None:
        if backend not in {"hdf5", "arrow"}:
            raise ValueError(f"Unsupported data backend: {backend}")
        self.path = path
        self.backend = backend
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def _dataset(self, split: str) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
        if self.backend == "hdf5":
            return SHUH5Dataset(self.path, split)
        return ArrowEEGDataset(self.path, split)

    def loaders(self) -> dict[str, DataLoader]:
        generator = torch.Generator().manual_seed(self.seed)
        loaders: dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            dataset = self._dataset(split)
            sampler = None
            shuffle = split == "train"
            if split == "train" and isinstance(dataset, ArrowEEGDataset):
                sampler = ArrowBlockShuffleSampler(dataset, seed=self.seed)
                shuffle = False
            loaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=seed_worker,
                generator=generator if split == "train" else None,
                drop_last=False,
            )
        return loaders
