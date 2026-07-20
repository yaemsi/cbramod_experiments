# Part 1 — Code Review, SHU-MI Reproduction, and Architecture Comparison

## Executive summary

Part 1 covers:

- **Task A:** review the original CBraMod preprocessing/data-loading code and implement improvements;
- **Task B:** reproduce CBraMod on SHU-MI using the released pretrained checkpoint and paper split;
- **Task C:** adapt EEGSimpleConv to the same SHU-MI examples and compare predictive and operational behavior.

The CBraMod reproduction is complete. The EEGSimpleConv implementation and comparison workflow are complete; its final five-seed metrics and benchmarks should be inserted after those runs finish.

## Task A — Code review

### Strengths of the original repository

- It provides preprocessing code for all downstream datasets used in the paper.
- SHU-MI preprocessing applies the expected 250-to-200 Hz conversion and four-second patch structure.
- LMDB avoids the worst case of storing one separate file per training example.
- The downstream code exposes pretrained fine-tuning and reports relevant binary-classification metrics.
- The repository and released checkpoint make approximate reproduction possible.

### Main weaknesses

#### Split assignment is implicit

The original SHU-MI script sorts filenames and slices the resulting list into train/validation/test ranges. This assumes filename order, number of sessions, and naming conventions remain exactly as expected. A missing or misnamed file can silently move a subject into the wrong split.

**Correction:** parse subject IDs explicitly, assign subjects 1–15/16–20/21–25, and assert that each subject appears in exactly one split.

#### Hard-coded paths and script-level execution

Machine-specific paths and global executable code make reuse, testing, and automation difficult.

**Correction:** configuration-driven paths, reusable functions, a CLI, and Makefile targets.

#### Limited validation and provenance

The original preprocessing does not persist a complete manifest or validate all assumptions before writing. It is difficult to establish which raw file produced a sample or which preprocessing version was used.

**Correction:** validate ranks, sizes, labels, finite values, subjects, sessions, and splits; persist source/provenance metadata; write resolved configurations and runtime manifests.

#### Inefficient LMDB writes

Opening/committing a transaction per example produces unnecessary synchronization and write overhead.

**Correction:** use batched/chunked writes. The reference implementation here uses HDF5; the scalable prototype uses compressed Arrow record batches and shards.

#### Unsafe normalization assumptions

The original loader applies a hard-coded division by 100 without documenting units in the stored representation.

**Correction:** make unit/scaling assumptions explicit and test MAT/EDF equivalence. The canonical schema records units and source format.

#### Validation and test shuffling

The released loader constructs validation and test loaders with shuffling enabled. Although aggregate metrics can remain unchanged, this adds needless nondeterminism and complicates debugging.

**Correction:** only the training loader shuffles.

#### Multiprocessing and portability

Opening storage handles during dataset construction can be fragile under different multiprocessing start methods.

**Correction:** lazily open HDF5/Arrow resources per worker/process and test worker-safe access.

#### Package and code quality

The provided scaffold initially had eager package imports that produced a circular-import error and omitted subpackages from wheel discovery.

**Correction:** keep package `__init__.py` files lightweight, use direct imports from defining modules, fix package discovery, and add import-order and build tests.

### Scalability concerns and recommended production changes

- avoid millions of small objects;
- partition processing by subject/recording;
- use restartable, idempotent preprocessing tasks;
- separate stable offline transforms from stochastic online augmentation;
- persist manifests, checksums, configuration hashes, and quality flags;
- materialize large sequential shards;
- prefetch and cache shards close to training nodes;
- monitor throughput at storage, decode, collation, and host-to-device stages.

The implemented Part 2 prototype demonstrates the reader/canonical-schema/sharded-storage boundary. See [`part2.md`](part2.md).

## Task B — Reproduce CBraMod on SHU-MI

### Dataset and protocol

The full processed dataset contains:

- 25 subjects and five sessions per subject;
- 11,988 four-second trials;
- 32 channels;
- original rate 250 Hz and 1,000 samples;
- resampled rate 200 Hz and 800 samples;
- binary left/right motor-imagery labels;
- subjects 1–15 for training, 16–20 for validation, and 21–25 for testing.

A strict audit runs before reported training and rejects missing subjects, incorrect counts/shapes, class omissions, overlap, or subject leakage.

### Model and optimization

The run uses:

- the released pretrained CBraMod backbone;
- verified checkpoint hash and strict state-dict compatibility;
- the all-patch-representations downstream classifier;
- 50 epochs and batch size 64;
- AdamW with weight decay 0.05;
- backbone learning rate `1e-4`;
- classifier learning rate `5e-4`;
- step-level cosine decay to `1e-6`;
- gradient clipping at norm 1.0;
- validation AUROC for model selection;
- final test evaluation only after checkpoint selection.

Metrics are balanced accuracy, AUROC, and trapezoidal AUC-PR. Average precision is retained separately as a diagnostic.

### Five-seed results

| Seed | Best epoch | Balanced accuracy | AUC-PR | AUROC |
|---:|---:|---:|---:|---:|
| 3407 | 5 | 0.6233 | 0.6983 | 0.6948 |
| 3408 | 26 | 0.5914 | 0.6635 | 0.6399 |
| 3409 | 1 | 0.6354 | 0.6986 | 0.7031 |
| 3410 | 3 | 0.6164 | 0.6851 | 0.6770 |
| 3411 | 8 | 0.6081 | 0.6772 | 0.6564 |

Aggregate comparison:

| Metric | Reproduction, mean ± population SD | Paper | Difference |
|---|---:|---:|---:|
| Balanced accuracy | **0.6149 ± 0.0148** | 0.6370 ± 0.0151 | -0.0221 |
| AUC-PR | **0.6845 ± 0.0133** | 0.7139 ± 0.0088 | -0.0294 |
| AUROC | **0.6742 ± 0.0235** | 0.6988 ± 0.0068 | -0.0246 |

The reproduction is in the same performance range as the paper, but it is not an exact numerical match. Mean metrics are approximately 2–3 absolute percentage points lower and AUROC is more variable. Seed 3409 reaches an AUROC of 0.7031, above the paper mean, which supports the correctness of the data, checkpoint, forward path, and evaluation implementation.

### Interpretation of the difference from the paper

The most likely causes, in descending order, are:

1. **seed sensitivity:** AUROC ranges from 0.6399 to 0.7031 and best epoch ranges from 1 to 26;
2. **held-out-subject variability:** validation and test contain only five different subjects each, and the average validation-to-test drop is substantial;
3. **different RNG consumption:** explicitly seeded workers/samplers do not recreate the authors' exact sample order or dropout sequence for the same integer seed;
4. **different PyTorch/CUDA kernels and versions:** Python 3.13 itself is unlikely to matter, but attention/matrix kernels, floating-point accumulation, and CUDA RNG implementations can;
5. **repository snapshot ambiguity:** the current public repository is not a frozen, dependency-pinned artifact of the paper experiments;
6. **minor numerical differences:** SciPy/NumPy resampling and floating-point order may alter optimization trajectories.

The metric definitions, subject split, dataset size, checkpoint, and main optimization settings were verified and are unlikely to explain the discrepancy.

### Overfitting observation

Four of five validation-selected checkpoints occur by epoch 8, while training loss continues to approach zero. This indicates rapid overfitting. Retaining the 50-epoch schedule is appropriate for paper alignment, but early stopping is useful for development and the EEGSimpleConv baseline.

### Reproduction artifacts

Each seed directory contains:

- `resolved_config.json`;
- `run.json`;
- `history.json`;
- `best_model.pt`;
- `metrics.json`.

The aggregate output is `outputs/cbramod_shu_mi/summary.json`. The repository does not embed the large run/checkpoint files.

## Task C — EEGSimpleConv comparison

### Implementation

The adapted EEGSimpleConv model follows the reference architecture:

- input `[batch, channels, time]`;
- internal temporal resampling;
- initial temporal convolution;
- two convolutional blocks with batch normalization, ReLU, and max pooling;
- exact width growth using `int(1.414 * width)`;
- global average pooling;
- binary classifier.

### Fair comparison protocol

Both models use exactly the same:

- SHU-MI materialized examples;
- subject splits;
- labels and amplitude scaling;
- metric functions;
- five seeds;
- validation-AUROC checkpoint selection;
- final-test evaluation policy.

The storage backend can be HDF5 or Arrow. A parity test verifies that both contain identical SHU-MI tensors and metadata. CBraMod only reshapes `[32, 800]` into `[32, 4, 200]`; EEGSimpleConv receives the original `[32, 800]` view.

The baseline uses its documented Adam and epoch-40 step-decay schedule. Euclidean alignment, test-time batch-normalization adaptation, mixup, and auxiliary subject classification are excluded because adding them only to EEGSimpleConv would compare complete pipelines rather than architectures.

### Measurements

The final comparison reports:

- five-seed balanced accuracy, AUC-PR, and AUROC;
- parameter count and trainable parameter count;
- checkpoint/state size;
- batch-1 mean/median/p95 latency;
- batch-64 throughput;
- peak GPU memory;
- training time when available.

Run:

```bash
make task-c
```

Expected generated artifacts:

```text
reports/task_c/
├── cbramod_benchmark.json
├── eegsimpleconv_benchmark.json
├── comparison.json
└── comparison.md
```

### Results status

The EEGSimpleConv code, tests, configurations, multi-seed runner, benchmarker, and report generator are complete. Final EEGSimpleConv numerical results are intentionally not stated here until the full runs finish. Once available, `reports/task_c/comparison.md` becomes the detailed evidence file and this section should be updated with its aggregate table.

### When to prefer each architecture

**Prefer CBraMod when:**

- transfer from large-scale pretrained EEG representations matters;
- labels are scarce;
- cross-dataset generalization is a priority;
- compute and memory budgets permit a transformer-style backbone;
- the deployment can benefit from a reusable foundation representation.

**Prefer EEGSimpleConv when:**

- low latency, memory, and operational simplicity dominate;
- training from scratch on one task is sufficient;
- deployment targets constrained devices;
- the baseline achieves comparable predictive performance;
- maintainability and fast iteration matter more than representation reuse.

A comparable EEGSimpleConv score would be scientifically important because it would question whether CBraMod's additional complexity is justified for this particular motor-imagery task, even if CBraMod remains more useful as a general foundation model.

## Quality and reproducibility checks

The suite covers preprocessing, audits, models, checkpoint loading, metrics, training, multi-seed aggregation, benchmarks, comparison generation, Arrow/HDF5 parity, and MAT/EDF equivalence. The current integrated repository previously completed:

```text
26 tests passed
Ruff checks passed
Pyright reported 0 errors and 0 warnings
Model smoke test passed
```

Exact commands and test groups are documented in [`../DESIGN.md`](../DESIGN.md).
