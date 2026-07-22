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
│   └── data_harmonization/         # schemas, readers, transforms, parallel engine, Arrow backends
├── tests/
├── notebooks/
├── reports/
│   ├── part1.md
│   └── part2.md
└── resources/data/                 # local SHU-MI/HBN fixtures or user-provided subsets
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
- `harmonize-shu`: materialize SHU-MI MAT or EDF/event data through the shared size-bundled parallel engine;
- `harmonize-bids`: materialize a BIDS-like SET/EDF/BDF subset through the same engine;
- `benchmark-streaming`: measure sequential iterable Arrow throughput;
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
- `SHUEdfReader`: reads continuous EDF and reconstructs trials from event TSV files. It is strict by default; optional lenient mode skips expected source-data failures and exposes a structured audit.

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

### 5.4 Shared size-bundled parallel engine

`data_harmonization/parallel.py` provides one orchestration implementation for SHU-MI MAT, SHU-MI EDF/event, and BIDS recordings. The unit of work is a deterministic bundle of recordings sized from source-byte estimates.

```text
source-specific discovery
        -> size-estimated recording bundles
        -> spawn-based worker processes
        -> worker-local Arrow shards + manifest fragments
        -> deterministic coordinator merge
        -> final manifest, shards, audit, and summary
```

Important properties:

- workers never write shared Arrow files; each owns a private bundle directory and keeps one writer open across many recordings;
- workers return only small result metadata, not large NumPy arrays;
- the coordinator/rank-0 process alone renders the `tqdm` progress bar;
- `spawn` is used instead of `fork`, avoiding inherited threaded-library state;
- results are merged in discovery order, so output is deterministic even when jobs finish out of order;
- BIDS recording identities preserve run/acquisition entities, preventing distinct recordings from producing the same window IDs;
- duplicate `sample_id` values and incompatible manifest schemas are rejected before any shard is published;
- strict mode writes an audit and aborts publication when any recording fails;
- lenient mode skips invalid recordings and records path, error type, and message;
- completed worker jobs are marked with `_SUCCESS.json`; interrupted strict runs retain `_work/` and can continue with `--resume`;
- success markers include a source/configuration fingerprint, so changed input files or preprocessing options invalidate stale worker output;
- finalization is a two-pass operation: first validate all fragments and build a shard plan, then publish;
- worker shards are moved with `shutil.move` into `_publishing_shards/` rather than copied, avoiding a second full corpus write when paths share a filesystem;
- `_publishing_shards/` is renamed to `shards/` only after every move succeeds; failed publication rolls shards back to their worker directories;
- a durable `_PUBLICATION_PLAN.json` maps final shard names to worker sources, allowing resume mode to recover an interrupted publication even after the coordinator process exits;
- manifest and summary files are written through temporary paths and atomically replaced;
- after successful publication, `_work/` is removed.

`num_workers=1` uses the same bundle-local/merge architecture serially. This makes serial-versus-parallel parity testable and avoids maintaining two code paths.

The coordinator writes end-to-end timing information to `summary.json`: processing, merge, and total wall time together with recordings/examples/signal MiB per second.

### 5.5 Parquet manifest and Arrow shards

The harmonized layout is:

```text
output_dir/
├── manifest.parquet
├── summary.json
├── source_audit.json
└── shards/
    ├── shard-00000.arrow
    └── ...
```

The manifest contains searchable metadata, including sample identity, dataset, subject/session/task, split, label, dimensions, channel metadata, shard path, record-batch index, and row index.

Dense float32 signals are stored in compressed Arrow IPC record batches. This avoids one-file-per-example overhead and supports sequential/memory-mapped reads. Batch and shard sizes are configurable through the Makefile.

### 5.6 Arrow training backends

`data_harmonization/datamodule.py` presents the same training interface as HDF5 and supports two Arrow modes:

- `arrow`: random access with record-batch-aware shuffling and a small decompressed-batch cache;
- `arrow_streaming`: iterable shard streaming with rank/worker partitioning, shard shuffling, and bounded-buffer example shuffling.

Both models receive `[channels, time]` from storage. Model-specific reshaping happens inside the model adapter:

- CBraMod: `[32, 800] -> [32, 4, 200]`;
- EEGSimpleConv: retains `[32, 800]` and resamples internally as configured by the architecture.

### 5.7 Backend parity

`data_harmonization/parity.py` compares HDF5 and Arrow by split and index, including:

- dataset lengths;
- labels;
- subjects and sessions;
- signal shapes;
- element-wise numerical values.

On the bundled real sample, maximum absolute signal difference is zero. MAT and EDF/event reconstruction produce identical labels and signal correlation above 0.999999 after unit alignment.

For large optional EDF corpora, the reader supports two policies:

- strict mode (default) aborts on the first invalid recording;
- lenient mode records the path, exception type, and message in `source_audit.json`, skips the complete recording atomically, and continues.

Corpus-level MAT/EDF tests pair recordings and examples by stable IDs, so one skipped EDF recording cannot shift all later comparisons.

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

`utils/data_benchmark.py` contains two complementary data-path benchmarks:

- `benchmark_streaming_dataset`: bounded-batch Arrow streaming measurement for quick tuning;
- `benchmark_dataloader_epoch`: one complete epoch through `EEGDataModule`, supporting `hdf5`, `arrow`, and `arrow_streaming`.

The full-epoch result records dataset/loader construction time, iterator startup, first-batch latency, total epoch time, observed versus expected examples, batches, payload bytes, examples/s, and signal MiB/s. Optional host-to-device transfer is included when a CUDA device is selected. The benchmark deliberately excludes model computation so data delivery can be measured independently.

`utils/compare.py` combines five-seed summaries and architecture benchmarks into JSON and Markdown under `outputs/results_models_comparison/`.

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
- model benchmark output;
- complete dataloader-epoch accounting and JSON output;
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
- SHU-MI MAT versus EDF/event equivalence;
- serial-versus-parallel manifest and tensor equivalence;
- worker failure auditing in strict and lenient modes;
- interrupted-run resume behavior;
- BIDS run-aware sample identity;
- duplicate-ID rejection before shard publication;
- move-based shard publication without `copy2`;
- rollback and resumability after a simulated publication failure;
- duplicate sample-ID protection;
- coordinator-only progress behavior by construction.

Run:

```bash
make test
make test-integration
make test-verbose
make check
```

## 9. Makefile reference

### Environment and quality

| Target | Purpose |
|---|---|
| `help` | List targets and common variables. |
| `sync` / `install` | Create/update the uv environment and install the project. |
| `lock` | Regenerate `uv.lock` without an explicit upgrade. |
| `update` | Upgrade permitted dependencies and sync. |
| `smoke` | CPU-friendly model/metric smoke test. |
| `test` | Run all tests quietly. |
| `test-verbose` | Run all tests verbosely. |
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
| `sample-preprocess` | Process the bundled subject/session sample. |
| `sample-inspect` | Inspect the intentionally incomplete bundled sample. |

### Harmonized data backend

| Target | Purpose |
|---|---|
| `harmonize-shu` | Parallel SHU-MI MAT harmonization with a coordinator/rank-0 progress bar. |
| `harmonize-shu-edf` | Parallel EDF/event harmonization; malformed recordings are skipped/audited by default. |
| `harmonize-hbn` | Parallel HBN/BIDS harmonization using whichever supported SET/EDF/BDF files are present. |
| `inspect-harmonized` | Summarize a manifest; strict mode can enforce the SHU protocol. |
| `compare-backends` | Verify numerical and metadata parity between HDF5 and Arrow. |
| `benchmark-streaming` | Measure a bounded number of iterable Arrow batches. |
| `benchmark-dataloader` | Measure one complete epoch through HDF5, random-access Arrow, or streaming Arrow. |
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
| `render-data-notebook` | Execute the SHU-MI exploration notebook and export HTML. |
| `explore-dataloader` | Open the harmonized dataloader benchmark notebook. |
| `render-dataloader-notebook` | Execute the full-epoch dataloader notebook and export HTML. |
| `clean` | Remove caches, builds, and generated experiment outputs. |
| `clean-data` | Remove generated processed/harmonized data while preserving raw data. |
| `distclean` | Run cleanup and remove `.venv`. |

### Common variables

| Variable | Default | Meaning |
|---|---|---|
| `RAW_DIR` | `resources/data/shu-mi/mat_files` | One authoritative SHU-MI MAT root. |
| `DATASET` | `outputs/data/preprocessed/shu-mi/shu_mi.h5` | HDF5 path or Arrow manifest, depending on backend. |
| `DATA_BACKEND` | `hdf5` | `hdf5`, `arrow`, or `arrow_streaming`. |
| `DATALOADER_DATA` | harmonized SHU manifest | Dataset path used by the full-epoch benchmark. |
| `DATALOADER_BACKEND` | `arrow_streaming` | Backend measured by `benchmark-dataloader`. |
| `DATALOADER_NUM_WORKERS` | `8` | Dataloader workers used for the epoch benchmark. |
| `DATALOADER_PREFETCH_FACTOR` | `4` | Batches prefetched per worker. |
| `CHECKPOINT` | empty | Optional local CBraMod checkpoint. |
| `OUTPUT_DIR` | `outputs` | Common generated-output root. |
| `SIMPLECONV_OUTPUT_DIR` | `outputs/eegsimpleconv_shu_mi` | EEGSimpleConv run root. |
| `REPRO_SEEDS` | `3407 3408 3409 3410 3411` | Multi-seed list. |
| `STRICT` | `1` | Enforce complete SHU protocol where supported. |
| `OVERWRITE` | `0` | Remove an existing harmonized output and rebuild. |
| `RESUME` | `0` | Reuse completed worker jobs from an interrupted harmonization run. |
| `HARMONIZE_WORKERS` | `4` | Worker processes operating on deterministic size-balanced recording bundles. |
| `SHU_TARGET_JOB_GIB` | `0.25` | Approximate SHU source GiB per bundle. |
| `HBN_TARGET_JOB_GIB` | `8` | Approximate HBN source GiB per bundle. |
| `HARMONIZE_MAX_RECORDINGS_PER_JOB` | `128` | Maximum recordings in one bundle. |
| `HARMONIZE_PROGRESS` | `1` | Enable coordinator/rank-0 `tqdm` progress. |
| `SHU_EDF_SKIP_INVALID` | `1` | Skip and audit malformed optional EDF/event recordings. |
| `HBN_SKIP_INVALID` | `1` | Skip and audit malformed heterogeneous BIDS recordings. |
| `HBN_ROOT` | `resources/data/hbn` | HBN/BIDS subset root. |
| `HBN_LIMIT_RECORDINGS` | empty | Optional maximum number of BIDS recordings. |
| `HBN_TARGET_RATE` | `auto` | Preserve native rate unless a numeric target is supplied. |
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
- preprocessing summary and end-to-end timing;
- source audit with processed/skipped/resumed recordings;
- dataset/source identity;
- sample-level provenance and preprocessing version;
- deterministic shard/manifest locations.

This allows a result to be traced back to the exact dataset representation, code/configuration, seed, and model artifact.

## 11. Known boundaries

- The public CBraMod repository is not a versioned snapshot of the exact paper environment; dependency and RNG differences can affect five-seed means.
- The HBN implementation is a focused prototype, not a complete implementation of every BIDS inheritance rule.
- HBN examples are unlabeled in this project and are not mixed with SHU-MI supervised metrics.
- Spatial interpolation to a universal montage, advanced artifact rejection, multi-node orchestration, object-store caching, and Kafka integration remain production extensions. Local bundle-level multiprocessing is implemented.
- The Arrow backend demonstrates local sharded loading; it does not claim a measured 1 GB/s distributed throughput target.

## 12. Third-party sources

This implementation draws on the publicly released CBraMod and EEGSimpleConv architectures and checkpoints. Their original licenses and repository notices remain applicable to the corresponding concepts and artifacts. The project does not redistribute private datasets or pretrained weights.
