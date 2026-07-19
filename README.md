# CBraMod vs EEGSimpleConv on SHU-MI

A reproducible implementation for the CBraMod take-home assessment. Both models
share the same SHU-MI preprocessing, subject split, data loader, training
infrastructure, and metric implementation.

## Environment

The project intentionally keeps Python 3.13 and the current PyTorch/CUDA stack
specified in `pyproject.toml`.

```bash
uv sync
make test
make smoke
```

## Reproduce CBraMod on SHU-MI

### 1. Prepare the data

Download and extract the password-protected SHU-MI archive. Point `RAW_DIR` at
the directory containing the `.mat` files; recursive discovery is supported.

```bash
make preprocess \
  RAW_DIR=/absolute/path/to/shu/mat_files \
  DATASET=data/processed/shu_mi.h5 \
  OVERWRITE=1
```

The preprocessor expects:

- `data`: `[trials, channels, time]`;
- `labels`: `1/2` or `0/1`;
- filenames containing explicit subject IDs such as `sub-001`.

It resamples each four-second trial from 1,000 to 800 points with
`scipy.signal.resample`, maps labels to `0/1`, and assigns subjects 1–15 to
train, 16–20 to validation, and 21–25 to test.

### 2. Refuse invalid reproduction data

```bash
make inspect-data DATASET=data/processed/shu_mi.h5
```

Strict inspection verifies all 25 subjects, exactly 11,988 examples, 32
channels, 800 points, both classes in every split, complete split coverage, and
no subject leakage. Training targets use strict inspection by default.

The repository includes one real subject-1 file only for pipeline validation:

```bash
make sample-preprocess
make sample-inspect
```

Warnings are expected for this incomplete sample; it cannot produce a reported
reproduction result.

### 3. Validate the released checkpoint

```bash
make check-checkpoint
```

This downloads the Hugging Face checkpoint, verifies its published SHA256, maps
the authors' state-dict names to the cleaned implementation, and requires a
complete architecture match.

A local checkpoint can be used instead:

```bash
make check-checkpoint CHECKPOINT=/absolute/path/pretrained_weights.pth
```

### 4. Run one seed

```bash
make train-cbramod \
  DATASET=data/processed/shu_mi.h5 \
  OUTPUT_DIR=outputs/cbramod_shu_mi
```

### 5. Run five seeds and aggregate

```bash
make reproduce-cbramod \
  DATASET=data/processed/shu_mi.h5 \
  OUTPUT_DIR=outputs/cbramod_shu_mi
```

Default seeds are `3407 3408 3409 3410 3411`. Override them with:

```bash
make reproduce-cbramod REPRO_SEEDS="1 2 3 4 5"
```

The final `summary.json` reports mean, sample standard deviation, range, every
individual run, the paper reference values, and the difference from the paper
mean.

## Paper-aligned CBraMod settings

`configs/cbramod.yaml` uses:

- 50 epochs and batch size 64;
- AdamW with weight decay 0.05;
- backbone LR `1e-4`;
- classifier LR `5e-4`;
- cosine annealing to `1e-6` after every optimizer step;
- gradient norm clipping at 1.0;
- full-precision training;
- the released pretrained backbone;
- the three-layer `all_patch_reps` classifier;
- checkpoint selection by validation AUROC.

AUC-PR is calculated with trapezoidal integration of the precision-recall
curve, matching the released evaluator. Average precision is retained as a
separate diagnostic metric.

See [`reports/shu_mi_reproduction.md`](reports/shu_mi_reproduction.md) for the
reference values, deviations, and reporting checklist.

## Outputs

Each seed writes:

```text
resolved_config.json
run.json
history.json
best_model.pt
metrics.json
```

The test split is evaluated only once after validation-based model selection.

## Development commands

```bash
make help
make test
make test-verbose
make smoke
make format
make lint
make typecheck
make check
```

## Task C: compare EEGSimpleConv with CBraMod

The comparison reuses the exact SHU-MI HDF5 file and subject split from Task B.
EEGSimpleConv consumes `[batch, 32, 800]` directly; its internal resampler
reduces 200 Hz signals to 80 Hz before the convolutional stack.

The implementation follows the public architecture defaults for cross-subject
motor-imagery decoding:

- 128 initial feature maps;
- two convolutional blocks;
- 80 Hz internal resampling;
- kernel size 8;
- ReLU activation;
- global temporal average pooling;
- one binary classification logit.

The controlled comparison deliberately does **not** add EEGSimpleConv's broader
training-pipeline enhancements such as Euclidean alignment, test-subject batch
normalization statistics, mixup, or a subject-classification auxiliary head.
Those techniques would change more than the architecture and would make it
harder to attribute differences to CBraMod pretraining versus the convolutional
baseline.

### Run five EEGSimpleConv seeds

```bash
make reproduce-simpleconv \
  DATASET=data/processed/shu_mi.h5 \
  SIMPLECONV_OUTPUT_DIR=outputs/eegsimpleconv_shu_mi
```

The baseline uses Adam for 50 epochs with learning rate `1e-3`, followed by a
10× decay at epoch 40. Checkpoints are still selected with validation AUROC and
are evaluated with the same balanced accuracy, AUC-PR, and AUROC code used for
CBraMod.

### Benchmark both architectures

Run the benchmarks on the same idle GPU. Random initialization is intentional:
weights do not affect architecture latency or parameter count, and this avoids
downloading the CBraMod checkpoint during profiling.

```bash
make benchmark-models \
  BENCHMARK_DEVICE=cuda \
  BENCHMARK_BATCHES="1 64" \
  BENCHMARK_WARMUP=20 \
  BENCHMARK_ITERATIONS=100
```

The benchmark JSON files contain parameter count, state size, mean/median/p95
latency, throughput, and peak allocated CUDA memory.

### Generate the final comparison

```bash
make compare-models \
  CBRAMOD_SUMMARY=outputs/cbramod_shu_mi/summary.json \
  SIMPLECONV_SUMMARY=outputs/eegsimpleconv_shu_mi/summary.json
```

This writes:

```text
reports/task_c/comparison.json
reports/task_c/comparison.md
```

The complete workflow can also be launched with `make task-c`, although running
seeds sequentially is recommended on a personal GPU to limit sustained heat.

## Explore the SHU-MI data

The executed notebook [`notebooks/shu_mi_data_exploration.ipynb`](notebooks/shu_mi_data_exploration.ipynb)
explains the raw MATLAB hierarchy, class and subject distributions, signal and
spectral characteristics, HDF5 preprocessing, and the model-specific tensor
views. It runs immediately on the bundled subject-1 sample.

```bash
make explore-data
```

To use the full archive, launch Jupyter with the data locations configured:

```bash
SHU_RAW_DIR=/absolute/path/to/shu/mat_files \
SHU_PROCESSED_PATH=data/processed/shu_mi.h5 \
make explore-data
```

An optional HBN section activates when `HBN_ROOT` points to a BIDS directory
containing `_eeg.set`, `_eeg.bdf`, or `_eeg.edf` files.
