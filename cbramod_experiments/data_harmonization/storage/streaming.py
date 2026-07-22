from __future__ import annotations

import math
import multiprocessing
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.ipc as ipc
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class _StreamRow:
    sample_id: str
    record_batch: int
    row_in_batch: int
    num_channels: int
    num_samples: int
    amplitude_scale: float
    label: int | None


class StreamingArrowEEGDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequential, rank-aware Arrow shard stream for large-scale training.

    The manifest is filtered once to select an experiment view. Signal shards are
    then opened one at a time and record batches are read sequentially. Shards are
    partitioned across distributed ranks and DataLoader workers without overlap.
    Approximate sample-level shuffling is provided by a bounded in-memory buffer,
    avoiding random reads across compressed shards.

    ``set_epoch`` should be called before each training epoch so shard order and
    buffered sample order change deterministically.
    """

    _MANIFEST_COLUMNS = [
        "sample_id",
        "shard_path",
        "record_batch",
        "row_in_batch",
        "num_channels",
        "num_samples",
        "amplitude_scale",
        "label",
    ]

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        sampling_rate_hz: float | None = None,
        num_channels: int | None = None,
        num_samples: int | None = None,
        require_labels: bool = False,
        shuffle_shards: bool = True,
        shuffle_buffer_size: int = 2048,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if world_size is not None and world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank is not None and world_size is not None and not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.split = split
        self.dataset_ids = tuple(dataset_ids or ())
        self.tasks = tuple(tasks or ())
        self.sampling_rate_hz = sampling_rate_hz
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.require_labels = require_labels
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.explicit_rank = rank
        self.explicit_world_size = world_size
        self._epoch_state = multiprocessing.Value("q", 0, lock=True)

        table = self._read_manifest_view()
        rows = table.to_pylist()
        if require_labels and any(row["label"] is None for row in rows):
            raise ValueError("Selected streaming view contains unlabeled examples")
        if not rows:
            raise ValueError("The selected streaming view contains no examples")

        grouped: dict[str, dict[int, list[_StreamRow]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            grouped[str(row["shard_path"])][int(row["record_batch"])].append(
                _StreamRow(
                    sample_id=str(row["sample_id"]),
                    record_batch=int(row["record_batch"]),
                    row_in_batch=int(row["row_in_batch"]),
                    num_channels=int(row["num_channels"]),
                    num_samples=int(row["num_samples"]),
                    amplitude_scale=float(row["amplitude_scale"]),
                    label=None if row["label"] is None else int(row["label"]),
                )
            )
        for batches in grouped.values():
            for batch_rows in batches.values():
                batch_rows.sort(key=lambda item: item.row_in_batch)

        self._rows_by_shard = {
            shard: dict(sorted(batches.items()))
            for shard, batches in sorted(grouped.items())
        }
        self._shards = tuple(self._rows_by_shard)
        self._num_examples = len(rows)

    def _read_manifest_view(self) -> pa.Table:
        dataset = pads.dataset(str(self.manifest_path), format="parquet")
        expression: pads.Expression | None = None

        def add_filter(new_filter: pads.Expression) -> None:
            nonlocal expression
            expression = new_filter if expression is None else expression & new_filter

        if self.split is not None:
            add_filter(pads.field("split") == self.split)
        if self.dataset_ids:
            add_filter(pads.field("dataset_id").isin(list(self.dataset_ids)))
        if self.tasks:
            add_filter(pads.field("task").isin(list(self.tasks)))
        if self.sampling_rate_hz is not None:
            add_filter(pads.field("sampling_rate_hz") == float(self.sampling_rate_hz))
        if self.num_channels is not None:
            add_filter(pads.field("num_channels") == int(self.num_channels))
        if self.num_samples is not None:
            add_filter(pads.field("num_samples") == int(self.num_samples))

        return dataset.to_table(
            columns=self._MANIFEST_COLUMNS,
            filter=expression,
        )

    def __len__(self) -> int:
        """Return the global number of selected examples.

        A distributed rank receives only its assigned shards. Use
        :meth:`estimated_examples_per_rank` for scheduler planning.
        """

        return self._num_examples

    def estimated_examples_per_rank(self) -> int:
        _, world_size = self._distributed_context()
        return math.ceil(self._num_examples / world_size)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state.value)

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch_state.value = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.set_epoch(int(state["epoch"]))

    def _distributed_context(self) -> tuple[int, int]:
        if self.explicit_rank is not None and self.explicit_world_size is not None:
            return self.explicit_rank, self.explicit_world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def assigned_shards(self) -> tuple[str, ...]:
        """Return the shards assigned to the current rank/worker for this epoch."""

        rank, world_size = self._distributed_context()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        global_worker_id = rank * num_workers + worker_id
        global_worker_count = world_size * num_workers

        shards = list(self._shards)
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch).shuffle(shards)
        return tuple(shards[global_worker_id::global_worker_count])

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rank, world_size = self._distributed_context()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        global_worker_id = rank * num_workers + worker_id
        global_worker_count = world_size * num_workers

        shards = list(self._shards)
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch).shuffle(shards)
        assigned = shards[global_worker_id::global_worker_count]

        stream = self._iter_assigned_shards(assigned)
        if self.shuffle_buffer_size <= 1:
            yield from stream
            return

        buffer_seed = self.seed + 1_000_003 * self.epoch + global_worker_id
        yield from _bounded_shuffle(
            stream,
            buffer_size=self.shuffle_buffer_size,
            rng=random.Random(buffer_seed),
        )

    def _iter_assigned_shards(
        self, shards: Iterable[str]
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for shard_path in shards:
            full_path = self.manifest_path.parent / shard_path
            if not full_path.exists():
                raise FileNotFoundError(full_path)
            source = pa.memory_map(str(full_path), "r")
            try:
                reader = ipc.RecordBatchFileReader(source)
                for batch_index, selected_rows in self._rows_by_shard[
                    shard_path
                ].items():
                    if batch_index >= reader.num_record_batches:
                        raise ValueError(
                            f"Manifest references batch {batch_index}, but "
                            f"{shard_path} has {reader.num_record_batches} batches"
                        )
                    batch = reader.get_batch(batch_index)
                    yield from self._decode_selected_rows(batch, selected_rows)
            finally:
                source.close()

    def _decode_selected_rows(
        self,
        batch: pa.RecordBatch,
        selected_rows: Sequence[_StreamRow],
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        signal_index = batch.schema.get_field_index("signal")
        sample_id_index = batch.schema.get_field_index("sample_id")
        if signal_index < 0 or sample_id_index < 0:
            raise ValueError("Arrow shard is missing signal/sample_id columns")
        signal_column = batch.column(signal_index)
        sample_id_column = batch.column(sample_id_index)

        for row in selected_rows:
            if row.row_in_batch >= batch.num_rows:
                raise ValueError(
                    f"Manifest row {row.row_in_batch} exceeds batch size {batch.num_rows}"
                )
            stored_id = sample_id_column[row.row_in_batch].as_py()
            if stored_id != row.sample_id:
                raise RuntimeError("Manifest and Arrow shard sample IDs do not match")
            payload = signal_column[row.row_in_batch].as_py()
            signal = np.frombuffer(payload, dtype=np.float32).copy()
            expected = row.num_channels * row.num_samples
            if signal.size != expected:
                raise ValueError(
                    f"Signal payload has {signal.size} values; expected {expected}"
                )
            signal = signal.reshape(row.num_channels, row.num_samples)
            signal /= row.amplitude_scale
            label_value = -1 if row.label is None else row.label
            yield torch.from_numpy(signal), torch.tensor(label_value, dtype=torch.long)


def _bounded_shuffle(
    stream: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    buffer_size: int,
    rng: random.Random,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sample in stream:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)
