from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from ..schema import CANONICAL_SCHEMA_VERSION, EEGWindow


_SIGNAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("signal", pa.large_binary(), nullable=False),
    ],
    metadata={b"schema_version": CANONICAL_SCHEMA_VERSION.encode()},
)

_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string()),
        pa.field("task", pa.string()),
        pa.field("split", pa.string()),
        pa.field("shard_path", pa.string(), nullable=False),
        pa.field("record_batch", pa.int32(), nullable=False),
        pa.field("row_in_batch", pa.int32(), nullable=False),
        pa.field("sampling_rate_hz", pa.float32(), nullable=False),
        pa.field("num_channels", pa.int32(), nullable=False),
        pa.field("num_samples", pa.int32(), nullable=False),
        pa.field("channel_names", pa.list_(pa.string()), nullable=False),
        pa.field("channel_mask", pa.list_(pa.bool_()), nullable=False),
        pa.field("label", pa.int32()),
        pa.field("start_seconds", pa.float64(), nullable=False),
        pa.field("duration_seconds", pa.float64(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("source_format", pa.string(), nullable=False),
        pa.field("preprocessing_version", pa.string(), nullable=False),
        pa.field("amplitude_scale", pa.float32(), nullable=False),
        pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
    ],
    metadata={b"schema_version": CANONICAL_SCHEMA_VERSION.encode()},
)


@dataclass(frozen=True)
class HarmonizationSummary:
    output_dir: str
    manifest_path: str
    examples: int
    shards: int
    datasets: list[str]
    source_formats: list[str]
    split_examples: dict[str, int]
    total_signal_bytes: int
    schema_version: str = CANONICAL_SCHEMA_VERSION
    source_audit: dict[str, Any] | None = None


class ArrowShardWriter:
    """Write canonical EEG windows to Arrow IPC shards plus a Parquet manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        records_per_batch: int = 256,
        batches_per_shard: int = 16,
        overwrite: bool = False,
    ) -> None:
        if records_per_batch <= 0 or batches_per_shard <= 0:
            raise ValueError("Batch and shard sizes must be positive")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output exists: {self.output_dir}; pass overwrite=True"
                )
            shutil.rmtree(self.output_dir)
        self.shards_dir = self.output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.records_per_batch = records_per_batch
        self.batches_per_shard = batches_per_shard
        self._pending: list[EEGWindow] = []
        self._manifest_rows: list[dict[str, Any]] = []
        self._shard_index = -1
        self._batch_index = 0
        self._sink: pa.NativeFile | None = None
        self._writer: ipc.RecordBatchFileWriter | None = None
        self._current_relative_path = ""
        self._total_signal_bytes = 0
        self._closed = False

    def add(self, window: EEGWindow) -> None:
        if self._closed:
            raise RuntimeError("Cannot add samples after closing the writer")
        window.validate()
        self._pending.append(window)
        if len(self._pending) >= self.records_per_batch:
            self._flush_batch()

    def add_all(self, windows: Iterable[EEGWindow]) -> None:
        for window in windows:
            self.add(window)

    def _open_shard(self) -> None:
        self._close_shard()
        self._shard_index += 1
        self._batch_index = 0
        shard_name = f"shard-{self._shard_index:05d}.arrow"
        self._current_relative_path = f"shards/{shard_name}"
        path = self.output_dir / self._current_relative_path
        self._sink = pa.OSFile(str(path), "wb")
        options = ipc.IpcWriteOptions(compression="zstd")
        self._writer = ipc.new_file(self._sink, _SIGNAL_SCHEMA, options=options)

    def _close_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def _flush_batch(self) -> None:
        if not self._pending:
            return
        if self._writer is None or self._batch_index >= self.batches_per_shard:
            self._open_shard()
        assert self._writer is not None

        signal_rows: list[dict[str, Any]] = []
        for row_in_batch, window in enumerate(self._pending):
            contiguous = np.ascontiguousarray(window.signal, dtype=np.float32)
            signal_bytes = contiguous.tobytes(order="C")
            self._total_signal_bytes += len(signal_bytes)
            signal_rows.append({"sample_id": window.sample_id, "signal": signal_bytes})
            self._manifest_rows.append(
                {
                    "sample_id": window.sample_id,
                    "dataset_id": window.dataset_id,
                    "subject_id": window.subject_id,
                    "session_id": window.session_id,
                    "task": window.task,
                    "split": window.split,
                    "shard_path": self._current_relative_path,
                    "record_batch": self._batch_index,
                    "row_in_batch": row_in_batch,
                    "sampling_rate_hz": float(window.sampling_rate_hz),
                    "num_channels": int(contiguous.shape[0]),
                    "num_samples": int(contiguous.shape[1]),
                    "channel_names": list(window.channel_names),
                    "channel_mask": window.channel_mask.astype(bool).tolist(),
                    "label": window.label,
                    "start_seconds": float(window.start_seconds),
                    "duration_seconds": float(window.duration_seconds),
                    "source_uri": window.source_uri,
                    "source_format": window.source_format,
                    "preprocessing_version": window.preprocessing_version,
                    "amplitude_scale": float(window.amplitude_scale),
                    "quality_flags": list(window.quality_flags),
                    "metadata_json": json.dumps(
                        window.metadata, sort_keys=True, default=str
                    ),
                }
            )

        table = pa.Table.from_pylist(signal_rows, schema=_SIGNAL_SCHEMA)
        batches = table.to_batches(max_chunksize=len(signal_rows))
        if len(batches) != 1:
            raise RuntimeError("Expected one Arrow record batch per pending batch")
        self._writer.write_batch(batches[0])
        self._batch_index += 1
        self._pending.clear()

    def close(self) -> HarmonizationSummary:
        if self._closed:
            raise RuntimeError("Writer has already been closed")
        self._flush_batch()
        self._close_shard()
        self._closed = True
        if not self._manifest_rows:
            raise ValueError("No EEG examples were written")

        manifest = pa.Table.from_pylist(self._manifest_rows, schema=_MANIFEST_SCHEMA)
        manifest_path = self.output_dir / "manifest.parquet"
        pq.write_table(
            manifest,
            manifest_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        split_examples: dict[str, int] = {}
        for row in self._manifest_rows:
            split = str(row["split"] or "unspecified")
            split_examples[split] = split_examples.get(split, 0) + 1
        summary = HarmonizationSummary(
            output_dir=str(self.output_dir),
            manifest_path=str(manifest_path),
            examples=len(self._manifest_rows),
            shards=self._shard_index + 1,
            datasets=sorted({str(row["dataset_id"]) for row in self._manifest_rows}),
            source_formats=sorted(
                {str(row["source_format"]) for row in self._manifest_rows}
            ),
            split_examples=split_examples,
            total_signal_bytes=self._total_signal_bytes,
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def __enter__(self) -> ArrowShardWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._close_shard()


class ArrowEEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Worker-safe random-access training dataset backed by Arrow shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None,
        *,
        require_labels: bool = True,
        max_cached_batches: int = 2,
    ) -> None:
        if split not in {None, "train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        if split is None:
            table = pq.read_table(self.manifest_path)
        else:
            table = pq.read_table(
                self.manifest_path,
                filters=[("split", "=", split)],
            )
        self.rows: list[dict[str, Any]] = table.to_pylist()
        self.split = split
        self.require_labels = require_labels
        self.max_cached_batches = max(1, max_cached_batches)
        self._readers: dict[
            str, tuple[pa.MemoryMappedFile, ipc.RecordBatchFileReader]
        ] = {}
        self._batch_cache: OrderedDict[tuple[str, int], pa.RecordBatch] = OrderedDict()
        if require_labels and any(row["label"] is None for row in self.rows):
            raise ValueError(
                f"Selected data contains unlabeled examples (split={split})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _reader(self, shard_path: str) -> ipc.RecordBatchFileReader:
        entry = self._readers.get(shard_path)
        if entry is None:
            full_path = self.manifest_path.parent / shard_path
            source = pa.memory_map(str(full_path), "r")
            reader = ipc.RecordBatchFileReader(source)
            self._readers[shard_path] = (source, reader)
            return reader
        return entry[1]

    def _batch(self, shard_path: str, batch_index: int) -> pa.RecordBatch:
        key = (shard_path, batch_index)
        batch = self._batch_cache.get(key)
        if batch is not None:
            self._batch_cache.move_to_end(key)
            return batch
        batch = self._reader(shard_path).get_batch(batch_index)
        self._batch_cache[key] = batch
        while len(self._batch_cache) > self.max_cached_batches:
            self._batch_cache.popitem(last=False)
        return batch

    def sample_metadata(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        batch = self._batch(str(row["shard_path"]), int(row["record_batch"]))
        signal_column = batch.column(batch.schema.get_field_index("signal"))
        sample_id_column = batch.column(batch.schema.get_field_index("sample_id"))
        row_index = int(row["row_in_batch"])
        if sample_id_column[row_index].as_py() != row["sample_id"]:
            raise RuntimeError("Manifest and Arrow shard sample IDs do not match")
        payload = signal_column[row_index].as_py()
        array = np.frombuffer(payload, dtype=np.float32).copy()
        expected = int(row["num_channels"]) * int(row["num_samples"])
        if array.size != expected:
            raise ValueError(
                f"Signal payload has {array.size} values; expected {expected}"
            )
        array = array.reshape(int(row["num_channels"]), int(row["num_samples"]))
        array /= float(row["amplitude_scale"])
        label = row["label"]
        if label is None and self.require_labels:
            raise ValueError(f"Example {row['sample_id']} has no label")
        label_value = -1 if label is None else int(label)
        return torch.from_numpy(array), torch.tensor(label_value, dtype=torch.long)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_batch_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        self._batch_cache.clear()
        for source, _ in self._readers.values():
            source.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()


class ArrowBlockShuffleSampler(torch.utils.data.Sampler[int]):
    """Shuffle Arrow record batches, then examples within each batch.

    Fully random sample order causes repeated decompression of distant Arrow
    record batches. This sampler preserves stochasticity while keeping reads
    mostly local to one compressed batch at a time.
    """

    def __init__(self, dataset: ArrowEEGDataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = {}
        for index, row in enumerate(dataset.rows):
            key = (str(row["shard_path"]), int(row["record_batch"]))
            groups.setdefault(key, []).append(index)
        self.groups = list(groups.values())

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        group_order = torch.randperm(len(self.groups), generator=generator).tolist()
        for group_index in group_order:
            group = self.groups[group_index]
            row_order = torch.randperm(len(group), generator=generator).tolist()
            for row_index in row_order:
                yield group[row_index]
