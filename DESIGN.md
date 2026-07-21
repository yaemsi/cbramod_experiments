# Technical Design

## 1. Goals and design principles

The project has four goals:

1. reproduce CBraMod fine-tuning on SHU-MI;
2. compare CBraMod and EEGSimpleConv fairly;
3. improve the reproducibility and scalability of the original code;
4. demonstrate a generalized data layer that supports SHU-MI and HBN/BIDS while remaining directly usable by training.

The main principles are:

- **one source of truth for examples:** both models consume the same processed signals and labels;
- **subject-level isolation:** train, validation, and test assignment occurs before training and is audited;
- **source/storage separation:** readers understand MAT, EDF, BDF, SET, and BIDS sidecars; trainers understand only tensors and labels;
- **non-destructive migration:** the validated HDF5 backend remains available while Arrow is introduced and tested for parity;
- **reproducibility by construction:** configurations, dataset audits, runtime metadata, checkpoint hashes, seeds, histories, and metrics are persisted;
- **small composable modules:** internal code imports defining modules directly, avoiding package-level circular imports.

## 2. Repository layout

```text
.
├── main.py                         # CLI entry point
├── Makefile                        # development and experiment workflows
├── configs/
│   ├── cbramod.yaml
│   └── eegsimpleconv.yaml
├── cbramod_experiments/
│   ├── datasets/                   # validated SHU-MI HDF5 path
│   ├── models/                     # CBraMod and EEGSimpleConv
│   ├── utils/                      # config, training, metrics, reproduction, comparison
│   └── data_harmonization/         # canonical schemas, readers, transforms, Arrow backend
├── tests/
├── notebooks/
├── reports/
│   ├── part1.md
│   └── part2.md
└── resources/data/shu-mi_dataset/  # local full archive; not committed
```

## 3. Command-line interface

`main.py` exposes the following operations:

- `preprocess`: build the reference SHU-MI HDF5 dataset from MAT files;
- `inspect-data`: audit HDF5 data and optionally enforce the full paper protocol;
- `train`: train and evaluate one configured model/seed;
- `reproduce`: run multiple seeds and aggregate results;
- `check-checkpoint`: download or validate the released CBraMod checkpoint;
- `benchmark`: measure parameter count, latency, throughput, and memory;
- `compare`: combine CBraMod and EEGSimpleConv metrics/benchmarks;
- `harmonize-shu`: materialize SHU-MI MAT or EDF/event data as Parquet plus Arrow;
- `harmonize-bids`: materialize a BIDS-like EDF/BDF/SET subset;
- `inspect-harmonized`: summarize and optionally audit a harmonized manifest;
- `compare-backends`: verify HDF5/Arrow sample parity;
- `smoke`: execute CPU-friendly model and metric checks.

Use `python -m main <command> --help` for command-specific arguments.

## 4. Reference SHU-MI data path

### 4.1 Raw representation

Each SHU-MI MAT session contains:

- `data`: trial tensor `[num_trials, 32, 1000]`;
- `labels`: left/right motor-imagery labels encoded as `1/2` or `0/1`;
- a filename containing explicit subject and session identifiers.

The original sampling rate is 250 Hz, so 1,000 samples represent four seconds.

### 4.2 `preprocess_shu`

Defined in `cbramod_experiments/datasets/shumi.py`, the preprocessing path:

1. discovers MAT files recursively;
2. parses subject and session IDs from filenames;
3. validates signal rank, channel count, label count, label vocabulary, and finite values;
4. resamples each trial from 1,000 to 800 samples using `scipy.signal.resample`;
5. converts labels to `0/1`;
6. assigns splits explicitly by subject:
   - subjects 1–15: train;
   - subjects 16–20: validation;
   - subjects 21–25: test;
7. writes signals, labels, subjects, sessions, source files, and split indices into HDF5.

The split is derived from parsed subject IDs rather than filename position.

### 4.3 Dataset auditing

The HDF5 audit verifies:

- expected shapes and dtypes;
- finite values;
- labels in `{0, 1}`;
- both classes in each split;
- subject-disjoint splits;
- complete, non-overlapping sample indices;
- expected subject groups;
- optionally, exactly 11,988 examples and all 25 subjects.

Strict audit is enabled by default for reported training runs. Debug/sample workflows can opt out.

### 4.4 HDF5 dataset and DataModule

The HDF5 dataset opens storage lazily to remain safe with multiprocessing workers. The shared `EEGDataModule` builds train, validation, and test loaders for either HDF5 or Arrow. Validation and test are never shuffled.

## 5. Harmonized data layer

### 5.1 Canonical schemas

`data_harmonization/schema.py` defines source-independent records:

- `EEGRecording`: continuous signal, sample rate, channel names/types, subject/session/task, events, units, source, and metadata;
- `EEGWindow`: fixed window plus channel mask, dataset identity, split/label, time range, quality information, and preprocessing version.

The canonical recording deliberately does not require 32 channels, 200 Hz, four-second duration, or labels. Those are dataset/task policies applied later.

### 5.2 Readers

`data_harmonization/readers/shu.py` implements:

- `SHUMatReader`: reads pre-segmented SHU-MI MAT trials;
- `SHUEdfReader`: reads continuous EDF and reconstructs trials from event TSV files.

`data_harmonization/readers/bids.py` implements a generalized BIDS-like reader for:

- EDF;
- BioSemi BDF;
- EEGLAB SET, with adjacent FDT when required;
- recording-level channel/event/EEG JSON sidecars;
- selected root/task-level metadata.

The reader normalizes channel types and excludes known non-EEG channels where metadata permits. Subject/task filters and recording limits allow small HBN subsets to be exercised without processing the full corpus.

### 5.3 Common transforms

`data_harmonization/transforms.py` provides:

- signal validation and finite-value checks;
- channel-name normalization;
- native-channel preservation or explicit channel selection;
- missing-channel masking when enabled;
- configurable resampling;
- event-based or sliding-window extraction;
- simple quality flags and provenance propagation.

Stable deterministic operations are materialized offline. Stochastic augmentation belongs in the training path and is intentionally not baked into stored examples.

### 5.4 Parquet manifest and Arrow shards

The harmonized layout is:

```text
output_dir/
├── manifest.parquet
├── summary.json
└── shards/
    ├── shard-00000.arrow
    └── ...
```

The manifest contains searchable metadata, including sample identity, dataset, subject/session/task, split, label, dimensions, channel metadata, shard path, record-batch index, and row index.

Dense float32 signals are stored in compressed Arrow IPC record batches. This avoids one-file-per-example overhead and supports sequential/memory-mapped reads. Batch and shard sizes are configurable through the Makefile.

### 5.5 Arrow training backend

`data_harmonization/datamodule.py` presents the same training interface as HDF5. A block-aware sampler shuffles record batches and then rows within each batch, preserving stochasticity without repeatedly decompressing unrelated batches.

Both models receive `[channels, time]` from storage. Model-specific reshaping happens inside the model adapter:

- CBraMod: `[32, 800] -> [32, 4, 200]`;
- EEGSimpleConv: retains `[32, 800]` and resamples internally as configured by the architecture.

### 5.6 Backend parity

`data_harmonization/parity.py` compares HDF5 and Arrow by split and index, including:

- dataset lengths;
- labels;
- subjects and sessions;
- signal shapes;
- element-wise numerical values.

On the bundled real sample, maximum absolute signal difference is zero. MAT and EDF/event reconstruction produce identical labels and signal correlation above 0.999999 after unit alignment.

## 6. Models

### 6.1 CBraMod

`models/cbramod.py` contains the cleaned downstream model implementation and checkpoint remapping logic. The loader:

- resolves the released Hugging Face checkpoint or a local path;
- verifies the known SHA256 when using the official artifact;
- maps original state-dict names to this implementation;
- rejects missing or incompatible backbone parameters;
- keeps the downstream classifier newly initialized.

The configured downstream classifier uses all patch representations. Training settings follow the released fine-tuning setup: batch size 64, 50 epochs, AdamW, weight decay 0.05, backbone LR `1e-4`, classifier LR `5e-4`, cosine decay, gradient clipping at 1.0, and validation-AUROC model selection.

### 6.2 EEGSimpleConv

`models/eegsimpleconv.py` adapts the reference architecture to the shared `[batch, channels, time]` input contract. It includes:

- resampling to the architecture's internal target frequency;
- temporal convolution blocks with batch normalization, ReLU, and max pooling;
- exact reference feature growth using `int(1.414 * width)`;
- global temporal average pooling;
- binary classification output.

Its configuration uses Adam for 50 epochs and the documented step decay at epoch 40. Optional pipeline-specific techniques from the original EEGSimpleConv study—Euclidean alignment, test-time batch-normalization adaptation, mixup, and auxiliary subject classification—are excluded so the submitted comparison focuses on architecture under a shared SHU-MI data path.

## 7. Training, evaluation, and reproduction

### 7.1 Training

`utils/train.py` implements:

- binary cross-entropy with logits;
- configurable Adam/AdamW optimizers;
- cosine or step learning-rate schedules;
- optional automatic mixed precision;
- gradient clipping;
- validation-driven checkpoint selection;
- optional early stopping;
- per-epoch history persistence;
- final test evaluation only after selecting the best validation checkpoint.

### 7.2 Metrics

`utils/metrics.py` reports:

- balanced accuracy at probability threshold 0.5;
- AUROC;
- AUC-PR using trapezoidal integration over the precision-recall curve, matching the released CBraMod evaluator;
- average precision as a separate diagnostic;
- loss and sample count.

### 7.3 Multi-seed reproduction

`utils/reproduce.py` executes independent seeds sequentially and writes:

```text
output_dir/
├── seed_<seed>/
│   ├── resolved_config.json
│   ├── run.json
│   ├── history.json
│   ├── best_model.pt
│   └── metrics.json
└── summary.json
```

The aggregate includes every run, mean, sample/population variability as applicable, range, best epochs, and paper reference values for CBraMod.

### 7.4 Benchmarking and comparison

`utils/benchmark.py` measures:

- total/trainable parameters;
- serialized state size;
- warm-up and timed inference latency;
- throughput;
- peak CUDA memory when available.

`utils/compare.py` combines five-seed summaries and architecture benchmarks into JSON and Markdown under `reports/results_models_comparison/`.

## 8. Tests

The suite covers the following groups.

### Package and configuration

- independent package import order and circular-import regression;
- configuration parsing and optimizer/scheduler construction;
- Python/PyTorch model factory behavior.

### Data and preprocessing

- subject/session filename parsing;
- subject-level split assignment;
- MAT shape and label validation;
- HDF5 preprocessing and audit behavior;
- full/incomplete protocol handling.

### Models and metrics

- CBraMod and EEGSimpleConv output shapes;
- exact EEGSimpleConv feature widths;
- checkpoint key remapping and hash verification;
- balanced accuracy, AUROC, AUC-PR, and average precision.

### Training and reporting

- one complete train/validation/test cycle;
- checkpoint selection and histories;
- multi-seed aggregation;
- benchmark output;
- final model-comparison report generation.

### Harmonization and storage

- canonical schema validation;
- Arrow round trips;
- worker-safe Arrow loading;
- block-aware shuffling;
- HDF5/Arrow exact parity;
- training through the Arrow backend;
- BIDS filename/sidecar discovery;
- EDF materialization;
- SHU-MI MAT versus EDF/event equivalence.

Run the self-contained suite with:

```bash
make test
make test-verbose
make check
```

Run real-file and full-manifest checks separately:

```bash
make test-integration SHU_ROOT=/absolute/path/to/shu-mi_dataset
```

The fast MAT harmonization tests generate one deterministic 100-trial fixture.
Real MAT/EDF equivalence stages only `sub-001`, session 01 from the configured
full archive. The complete 11,988-example expectation is enforced only by the
strict full-manifest integration audit.

## 9. Makefile reference

### Environment and quality

| Target | Purpose |
|---|---|
| `help` | List targets and common variables. |
| `sync` / `install` | Create/update the uv environment and install the project. |
| `lock` | Regenerate `uv.lock` without an explicit upgrade. |
| `update` | Upgrade permitted dependencies and sync. |
| `smoke` | CPU-friendly model/metric smoke test. |
| `test` | Run fast tests, excluding real-data integration checks. |
| `test-integration` | Run tests requiring real SHU-MI files or a full manifest. |
| `test-all` | Run fast and integration tests together. |
| `test-verbose` | Run fast tests verbosely. |
| `lint` / `lint-fix` | Run Ruff checks, optionally applying safe fixes. |
| `format` / `format-check` | Apply or verify Ruff formatting. |
| `typecheck` | Run Pyright. |
| `check` / `ci` / `all` | Run formatting, lint, type checks, tests, and smoke tests. |
| `hooks` | Install pre-commit hooks. |

### SHU-MI reference backend

| Target | Purpose |
|---|---|
| `preprocess` | Convert SHU-MI MAT files to HDF5. |
| `inspect-data` | Audit HDF5 data and, by default, enforce the paper protocol. |
| `stage-sample` | Copy one configured subject/session from the full local archive. |
| `sample-preprocess` | Process the staged 100-trial subject/session sample. |
| `sample-inspect` | Inspect the intentionally incomplete staged sample. |

### Harmonized data backend

| Target | Purpose |
|---|---|
| `harmonize-shu` | Build Parquet/Arrow from SHU-MI MAT. |
| `harmonize-shu-edf` | Build Parquet/Arrow by reconstructing SHU-MI trials from EDF plus events. |
| `harmonize-hbn` | Build Parquet/Arrow from an HBN/BIDS EDF/BDF/SET subset. |
| `inspect-harmonized` | Summarize a manifest; strict mode can enforce the SHU protocol. |
| `compare-backends` | Verify numerical and metadata parity between HDF5 and Arrow. |
| `sample-harmonize` | Build Arrow from the bundled MAT sample. |
| `sample-harmonize-edf` | Build Arrow from the bundled EDF/event sample. |
| `sample-compare-backends` | Run bundled HDF5/Arrow parity validation. |
| `sample-harmonize-bids` | Exercise the generic BIDS reader on the bundled EDF sample. |

### Models and experiments

| Target | Purpose |
|---|---|
| `check-checkpoint` | Download/validate the CBraMod checkpoint and hash. |
| `train-cbramod` | Run one CBraMod seed. |
| `reproduce-cbramod` | Run and aggregate the configured five CBraMod seeds. |
| `reproduce-cbramod-debug` | Multi-seed run that permits incomplete data. |
| `train-simpleconv` | Run one EEGSimpleConv seed. |
| `reproduce-simpleconv` | Run and aggregate five EEGSimpleConv seeds. |
| `benchmark-cbramod` | Benchmark CBraMod. |
| `benchmark-simpleconv` | Benchmark EEGSimpleConv. |
| `benchmark-models` | Benchmark both models. |
| `compare-models` | Produce combined metric/efficiency comparison files. |
| `task-c` | Run SimpleConv reproduction, both benchmarks, and comparison. |

### Exploration and cleanup

| Target | Purpose |
|---|---|
| `explore-data` | Open the SHU-MI exploration notebook in JupyterLab. |
| `render-data-notebook` | Execute the notebook and export HTML. |
| `clean` | Remove caches, builds, and generated experiment outputs. |
| `clean-data` | Remove generated processed/harmonized data while preserving raw data. |
| `distclean` | Run cleanup and remove `.venv`. |

### Common variables

| Variable | Default | Meaning |
|---|---|---|
| `RAW_DIR` | `data/raw/shu` | SHU-MI MAT root. |
| `DATASET` | `data/processed/shu_mi.h5` | HDF5 path or Arrow manifest, depending on backend. |
| `DATA_BACKEND` | `hdf5` | `hdf5` or `arrow`. |
| `CHECKPOINT` | empty | Optional local CBraMod checkpoint. |
| `OUTPUT_DIR` | `outputs/cbramod_shu_mi` | CBraMod run root. |
| `SIMPLECONV_OUTPUT_DIR` | `outputs/eegsimpleconv_shu_mi` | EEGSimpleConv run root. |
| `REPRO_SEEDS` | `3407 3408 3409 3410 3411` | Multi-seed list. |
| `STRICT` | `1` | Enforce complete SHU protocol where supported. |
| `OVERWRITE` | `0` | Permit replacing generated data. |
| `HBN_ROOT` | `data/raw/hbn_subset` | HBN/BIDS subset root. |
| `HBN_LIMIT_RECORDINGS` | `3` | Maximum BIDS recordings in the POC. |
| `HBN_TARGET_RATE` | `200` | Optional HBN target sample rate. |
| `HBN_WINDOW_SECONDS` | `4` | HBN window duration. |
| `HBN_STRIDE_SECONDS` | `4` | HBN window stride. |
| `ARROW_RECORDS_PER_BATCH` | `256` | Rows per Arrow record batch. |
| `ARROW_BATCHES_PER_SHARD` | `16` | Record batches per Arrow shard. |
| `BENCHMARK_DEVICE` | `auto` | Benchmark device. |
| `PYTEST_ARGS` | empty | Extra arguments passed to pytest. |

## 10. Output and provenance contracts

Every reported seed stores:

- the fully resolved experiment configuration;
- Python, PyTorch, CUDA, device, dataset audit, model, parameter count, and checkpoint metadata;
- per-epoch training/validation history;
- validation-selected checkpoint;
- final validation and test metrics.

Harmonized datasets store:

- the manifest and shard layout;
- preprocessing summary;
- dataset/source identity;
- sample-level provenance and preprocessing version.

This allows a result to be traced back to the exact dataset representation, code/configuration, seed, and model artifact.

## 11. Known boundaries

- The public CBraMod repository is not a versioned snapshot of the exact paper environment; dependency and RNG differences can affect five-seed means.
- The HBN implementation is a focused prototype, not a complete implementation of every BIDS inheritance rule.
- HBN examples are unlabeled in this project and are not mixed with SHU-MI supervised metrics.
- Spatial interpolation to a universal montage, advanced artifact rejection, distributed preprocessing, object-store caching, and Kafka orchestration are production extensions rather than take-home requirements.
- The Arrow backend demonstrates local sharded loading; it does not claim a measured 1 GB/s distributed throughput target.

## 12. Third-party sources

This implementation draws on the publicly released CBraMod and EEGSimpleConv architectures and checkpoints. Their original licenses and repository notices remain applicable to the corresponding concepts and artifacts. The project does not redistribute private datasets or pretrained weights.
