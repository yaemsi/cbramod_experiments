# CBraMod Homework: Reproduction, Model Comparison, and EEG Data Harmonization

This repository implements the CBraMod take-home assessment:

- review and improve the original CBraMod data pipeline;
- reproduce CBraMod on SHU-MI using the released pretrained checkpoint;
- compare CBraMod with EEGSimpleConv on exactly the same examples;
- design and prototype a scalable harmonization/streaming layer for heterogeneous EEG sources.

The paper-aligned SHU-MI path remains available as HDF5. A second backend stores canonical EEG examples in compressed Arrow IPC shards indexed by a Parquet manifest and can be consumed by the same training code.

## Documentation

- [`DESIGN.md`](DESIGN.md): code architecture, principal functions, tests, CLI, Makefile targets, output contracts, and implementation boundaries.
- [`reports/part1.md`](reports/part1.md): Task A code review, Task B reproduction, and Task C model comparison.
- [`reports/part2.md`](reports/part2.md): large-scale harmonization design and the implemented parallel prototype.
- [`notebooks/shu_mi_data_exploration.ipynb`](notebooks/shu_mi_data_exploration.ipynb): SHU-MI data exploration.
- [`notebooks/harmonized_dataloader_benchmark.ipynb`](notebooks/harmonized_dataloader_benchmark.ipynb): load harmonized Arrow data and time a complete dataloader epoch.

## Environment

The project uses the Python 3.13 and PyTorch/CUDA stack declared in `pyproject.toml`.

```bash
uv sync
make check
```

Discover available commands with:

```bash
make help
uv run python -m main --help
```

## Project workflows

### 1. Build the SHU-MI HDF5 reference dataset

Point `RAW_DIR` to one authoritative MAT tree. Do not point it to a parent containing duplicate extracted copies of the same subject/session.

```bash
make preprocess \
  RAW_DIR=/absolute/path/to/shu/mat_files \
  DATASET=outputs/data/preprocessed/shu-mi/shu_mi.h5 \
  OVERWRITE=1

make inspect-data
```

Strict inspection checks the paper protocol: 25 subjects, 11,988 examples, 32 channels, 800 samples, valid labels, complete/disjoint splits, and no subject leakage.

### 2. Reproduce CBraMod

```bash
make check-checkpoint
make reproduce-cbramod
```

A local checkpoint can be supplied with `CHECKPOINT=/path/to/pretrained_weights.pth`.

### 3. Run EEGSimpleConv and generate Task C results

```bash
make reproduce-simpleconv
make benchmark-models BENCHMARK_DEVICE=cuda
make compare-models
```

Or run the complete sequence:

```bash
make task-c
```

### 4. Build the parallel harmonized SHU-MI backend

The same parallel engine is used by SHU-MI MAT, SHU-MI EDF/event, and HBN/BIDS sources. Recordings are deterministically bundled by estimated source size; each worker keeps one Arrow writer open across a bundle, so final shards contain many recordings rather than one file per recording. Only rank 0 merges outputs and owns the `tqdm` progress bar.

```bash
make harmonize-shu \
  RAW_DIR=/absolute/path/to/shu/mat_files \
  HARMONIZE_WORKERS=4 \
  OVERWRITE=1
```

Example progress output:

```text
Harmonizing shu-mat:  63%|██████████████▍        | 79/125 [01:14<00:42, 1.08 recording/s]
```

The result is written to:

```text
outputs/data/harmonized/shu_mi/
├── manifest.parquet
├── summary.json
├── source_audit.json
└── shards/
    ├── shard-00000.arrow
    └── ...
```

Validate the output and compare it with HDF5:

```bash
make inspect-harmonized
make compare-backends
make test-integration
```

Train directly from the random-access Arrow backend:

```bash
make train-cbramod \
  DATASET=outputs/data/harmonized/shu_mi/manifest.parquet \
  DATA_BACKEND=arrow
```

The streaming training backend is also available:

```bash
make train-cbramod \
  DATASET=outputs/data/harmonized/shu_mi/manifest.parquet \
  DATA_BACKEND=arrow_streaming
```

Measure one complete data-only epoch through the same dataloader abstraction:

```bash
make benchmark-dataloader \
  DATALOADER_DATA=outputs/data/harmonized/shu_mi/manifest.parquet \
  DATALOADER_BACKEND=arrow_streaming \
  DATALOADER_NUM_WORKERS=4 \
  DATALOADER_DEVICE=cpu
```

The command writes `outputs/benchmarks/dataloader_epoch.json` and reports loader construction, first-batch latency, full-epoch time, examples/s, and uncompressed signal MiB/s. The notebook provides an interactive walkthrough and an optional worker-count sweep:

```bash
make explore-dataloader
```

### 5. Harmonize SHU-MI EDF plus events

The EDF path is an optional continuous-recording validation path; reported model results use MAT files. Known malformed EDF/event recordings are skipped by default and recorded in `source_audit.json`.

```bash
make harmonize-shu-edf \
  HARMONIZE_WORKERS=4 \
  OVERWRITE=1
```

Force strict behavior with:

```bash
make harmonize-shu-edf SHU_EDF_SKIP_INVALID=0 OVERWRITE=1
```

### 6. Harmonize an HBN/BIDS subset

The reader discovers whichever supported EEG files are actually present (`.set`/`.fdt`, `.edf`, or `.bdf`) and reads applicable sidecars. It does not require every format to exist.

```bash
make harmonize-hbn \
  HBN_ROOT=/absolute/path/to/hbn_subset \
  HBN_LIMIT_RECORDINGS=10 \
  HBN_TARGET_RATE=200 \
  HBN_WINDOW_SECONDS=4 \
  HBN_STRIDE_SECONDS=4 \
  HARMONIZE_WORKERS=4 \
  OVERWRITE=1
```

HBN examples remain separate from the supervised SHU-MI comparison.

## Parallel harmonization controls

| Variable | Default | Meaning |
|---|---:|---|
| `HARMONIZE_WORKERS` | `4` | Worker processes that handle size-balanced recording bundles. |
| `HARMONIZE_PROGRESS` | `1` | Set to `0` to disable the coordinator/rank-0 `tqdm` bar. |
| `SHU_TARGET_JOB_GIB` | `0.25` | Approximate SHU source GiB per worker bundle. |
| `HBN_TARGET_JOB_GIB` | `8` | Approximate HBN source GiB per worker bundle. |
| `HARMONIZE_MAX_RECORDINGS_PER_JOB` | `128` | Safety cap on recordings packed into one bundle. |
| `RESUME` | `0` | Reuse completed worker outputs left by an interrupted run. |
| `OVERWRITE` | `0` | Remove an existing output and rebuild from scratch. |
| `SKIP_INVALID` | `0` | Lenient mode for SHU-MI MAT. |
| `SHU_EDF_SKIP_INVALID` | `1` | Lenient mode for the optional EDF/event corpus. |
| `HBN_SKIP_INVALID` | `1` | Lenient mode for heterogeneous BIDS subsets. |
| `ARROW_RECORDS_PER_BATCH` | `256` | Examples per compressed Arrow record batch. |
| `ARROW_BATCHES_PER_SHARD` | `16` | Record batches per worker-local shard. |

Bundle sizes are estimated from source bytes; EEGLAB `.set` estimates also include a sibling `.fdt` file when present. `OVERWRITE=1` and `RESUME=1` are mutually exclusive. On success, temporary `_work/` directories are removed. On a strict failure or interruption, completed job directories remain and can be reused with `RESUME=1`. A source/configuration fingerprint prevents stale worker output from being reused after an input file or preprocessing setting changes.

The Makefile constrains common native numerical thread pools to one thread per worker to avoid process × BLAS/FFT oversubscription.

## Testing and quality

```bash
make test             # self-contained tests
make test-integration # local real-data tests
make format-check
make lint
make typecheck
make smoke
make check
```

The tests cover preprocessing, model behavior, metrics, training, checkpoint loading, Arrow/HDF5 parity, streaming, MAT/EDF reconstruction, and serial-versus-parallel harmonization equivalence.

## Main Makefile targets

| Target | Purpose |
|---|---|
| `sync` | Install/synchronize the environment. |
| `check` | Formatting, lint, type checks, unit tests, and smoke test. |
| `preprocess` | Build the SHU-MI HDF5 reference backend. |
| `inspect-data` | Audit the HDF5 dataset. |
| `harmonize-shu` | Parallel MAT-to-Arrow/Parquet harmonization. |
| `harmonize-shu-edf` | Parallel EDF/event harmonization with structured failure audit. |
| `harmonize-hbn` | Parallel BIDS harmonization through the same engine. |
| `inspect-harmonized` | Summarize/audit a harmonized manifest. |
| `compare-backends` | Check HDF5/Arrow numerical and metadata parity. |
| `benchmark-streaming` | Measure a bounded number of iterable Arrow batches. |
| `benchmark-dataloader` | Measure one complete HDF5/Arrow dataloader epoch. |
| `reproduce-cbramod` | Run the five CBraMod seeds. |
| `reproduce-simpleconv` | Run the five EEGSimpleConv seeds. |
| `task-c` | Run the model comparison workflow. |
| `help` | Show all targets and configurable variables. |

See [`DESIGN.md`](DESIGN.md#makefile-reference) for the full reference.
