# CBraMod Homework: SHU-MI Reproduction, Model Comparison, and EEG Data Harmonization

This repository contains the implementation for the CBraMod take-home assessment:

- review and improve the original CBraMod data pipeline;
- reproduce CBraMod results on SHU-MI using the released pretrained weights;
- compare CBraMod with EEGSimpleConv under a shared evaluation protocol;
- prototype a generalized EEG ingestion and streaming pipeline for SHU-MI and HBN/BIDS data.

The validated SHU-MI path is preserved in HDF5, while an integrated Parquet/Arrow backend demonstrates how the same training code can consume harmonized data from heterogeneous EEG sources.

## Documentation

- [`DESIGN.md`](DESIGN.md): package architecture, principal functions, data contracts, tests, outputs, and Makefile reference.
- [`reports/part1.md`](reports/part1.md): code review, CBraMod reproduction, and EEGSimpleConv comparison protocol/results.
- [`reports/part2.md`](reports/part2.md): large-scale data-harmonization design and implemented prototype.
- [`notebooks/shu_mi_data_exploration.ipynb`](notebooks/shu_mi_data_exploration.ipynb): executable exploration of raw and processed SHU-MI data.

## Environment

The project intentionally uses the Python 3.13 and current PyTorch/CUDA stack declared in `pyproject.toml`.

```bash
uv sync
make check
```

Useful discovery commands:

```bash
make help
python -m main --help
```

## Quick start

### 1. Preprocess and validate full SHU-MI

`RAW_DIR` may point directly to the MAT directory or to a parent directory; discovery is recursive.

```bash
make preprocess \
  RAW_DIR=/absolute/path/to/shu/mat_files \
  DATASET=data/processed/shu_mi.h5 \
  OVERWRITE=1

make inspect-data DATASET=data/processed/shu_mi.h5
```

Strict validation checks the paper protocol: 25 subjects, 11,988 examples, 32 channels, 800 samples per trial, class coverage, subject-disjoint splits, and complete sample coverage.

### 2. Validate the CBraMod checkpoint

```bash
make check-checkpoint
```

To use a local checkpoint:

```bash
make check-checkpoint CHECKPOINT=/absolute/path/pretrained_weights.pth
```

### 3. Train or reproduce CBraMod

One seed:

```bash
make train-cbramod \
  DATASET=data/processed/shu_mi.h5 \
  DATA_BACKEND=hdf5
```

Five seeds and aggregation:

```bash
make reproduce-cbramod \
  DATASET=data/processed/shu_mi.h5 \
  DATA_BACKEND=hdf5
```

Default seeds are `3407 3408 3409 3410 3411`. Each run writes its resolved configuration, runtime manifest, training history, best checkpoint, and final metrics. The aggregate is written to `outputs/cbramod_shu_mi/summary.json`.

### 4. Run EEGSimpleConv and compare models

```bash
make reproduce-simpleconv \
  DATASET=data/processed/shu_mi.h5 \
  DATA_BACKEND=hdf5

make benchmark-models BENCHMARK_DEVICE=cuda
make compare-models
```

Or run the complete sequence:

```bash
make task-c
```

### 5. Build and use the harmonized Arrow backend

Build Arrow shards and a Parquet manifest from the same SHU-MI MAT files:

```bash
make harmonize-shu \
  RAW_DIR=/absolute/path/to/shu/mat_files \
  HARMONIZED_SHU_DIR=data/harmonized/shu_mi \
  OVERWRITE=1
```

Validate exact parity with the HDF5 reference:

```bash
make compare-backends \
  DATASET=data/processed/shu_mi.h5 \
  HARMONIZED_SHU_MANIFEST=data/harmonized/shu_mi/manifest.parquet
```

Train directly from Arrow without changing model code:

```bash
make train-cbramod \
  DATASET=data/harmonized/shu_mi/manifest.parquet \
  DATA_BACKEND=arrow

make train-simpleconv \
  DATASET=data/harmonized/shu_mi/manifest.parquet \
  DATA_BACKEND=arrow
```

### 6. Harmonize a small HBN/BIDS subset

Preserve the selected subjects' BIDS directory hierarchy and sidecars. The reader supports EDF, BDF, and EEGLAB SET/FDT recordings.

```bash
make harmonize-hbn \
  HBN_ROOT=/absolute/path/to/hbn_subset \
  HBN_LIMIT_RECORDINGS=3 \
  HBN_TARGET_RATE=200 \
  HBN_WINDOW_SECONDS=4 \
  HBN_STRIDE_SECONDS=4 \
  OVERWRITE=1
```

The HBN path validates generalized ingestion. HBN examples are not mixed into the supervised SHU-MI motor-imagery comparison.

## Local data, staged sample, and notebook

Large EEG files are not committed to the repository. By default, the full
SHU-MI archive is expected below:

```text
resources/data/shu-mi_dataset/
├── mat_files/
├── edf_files/
├── events/
└── preprocessed/        # generated
```

Override the location with `SHU_ROOT=/absolute/path/to/shu-mi_dataset` in
Makefile commands or `SHU_MI_ROOT` when invoking pytest directly.

The sample workflows first stage only `sub-001`, session 01 from the full local
archive. They therefore remain 100-trial checks even when the source directory
contains all 11,988 examples:

```bash
make stage-sample
make sample-compare-backends
make sample-harmonize-edf
make sample-harmonize-bids
```

Launch the data-exploration notebook:

```bash
make explore-data
```


## Testing

Fast tests are self-contained and do not scan the complete local EEG archive:

```bash
make test
```

Tests that compare real MAT and EDF files or audit a complete generated manifest
are marked as integration tests:

```bash
make test-integration SHU_ROOT=/absolute/path/to/shu-mi_dataset
```

To audit a non-default full Arrow manifest directly:

```bash
SHU_MI_MANIFEST=/absolute/path/to/manifest.parquet uv run pytest -m integration
```

## Main Makefile workflows

| Workflow | Command |
|---|---|
| Install environment | `make sync` |
| Run all fast quality checks | `make check` |
| Run real-data integration tests | `make test-integration` |
| Preprocess SHU-MI to HDF5 | `make preprocess` |
| Audit SHU-MI | `make inspect-data` |
| Validate/download checkpoint | `make check-checkpoint` |
| Run one CBraMod seed | `make train-cbramod` |
| Run five CBraMod seeds | `make reproduce-cbramod` |
| Run five EEGSimpleConv seeds | `make reproduce-simpleconv` |
| Benchmark both models | `make benchmark-models` |
| Build final model comparison | `make compare-models` |
| Build SHU-MI Arrow backend | `make harmonize-shu` |
| Compare HDF5 and Arrow | `make compare-backends` |
| Harmonize HBN subset | `make harmonize-hbn` |
| Show every target and variable | `make help` |

The complete target-by-target reference is in [`DESIGN.md`](DESIGN.md#makefile-reference).
