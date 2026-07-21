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

#### General maintainability and configuration

1. **Duplicated dataset-specific workflows**

   The repository contains substantial duplication across dataset-specific
   preprocessing, training, and evaluation scripts. Similar control flow is
   reimplemented with small dataset-dependent changes, increasing maintenance
   cost and making it easier for fixes or behavioral changes to be applied
   inconsistently.

   **Correction:** isolate dataset-specific behavior behind shared reader and
   dataset interfaces, reuse common trainer and evaluator components, and use
   configuration-driven factories to construct datasets, models, and tasks.
   Inheritance may be appropriate for some source adapters, but composition and
   explicit interfaces are preferable for most of the shared pipeline.

2. **Hard-coded configuration and implicit device placement**

   Paths, dimensions, preprocessing assumptions, and training parameters are
   frequently embedded directly in scripts. Several components also invoke
   CUDA-specific operations directly instead of receiving an explicit device.
   This couples the code to particular machines, complicates CPU execution, and
   makes multi-GPU or distributed execution fragile.

   **Correction:** represent runtime and experiment settings using validated
   configuration objects, expose user-controlled values through configuration
   files or CLI arguments, and resolve device placement once at the application
   boundary. Models and trainers should use an explicit `torch.device` rather
   than invoking `.cuda()` internally.

3. **Scattered execution and orchestration logic**

   Architecture definitions, command-line parsing, demonstration code, and
   training orchestration are spread across several modules. Separate commands
   for preprocessing, training, and evaluation are reasonable, but their
   behavior and configuration should be consistent.

   **Correction:** expose a centralized CLI with explicit subcommands, while
   keeping model modules focused on reusable architecture definitions and
   keeping orchestration in dedicated application modules.


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

### Problem setting and model inputs

SHU-MI is a binary motor-imagery EEG classification task. During each trial, a
participant imagines moving either the left or the right hand, and the model
must infer the imagined movement from four seconds of multichannel EEG.

Each raw trial contains 32 channels sampled at 250 Hz:

```text
raw signal:       [32 channels, 1000 samples]
duration:         4 seconds
original labels:  1 = left, 2 = right
encoded labels:   0 = left, 1 = right
```

The signals are resampled to 200 Hz, preserving the four-second duration:

```text
[32, 1000] at 250 Hz
        ↓
[32, 800] at 200 Hz
```

The split is subject-wise rather than trial-wise:

```text
training:   subjects 1–15
validation: subjects 16–20
test:       subjects 21–25
```

This evaluates generalization to people whose EEG was never observed during
training. Both architectures produce one binary logit per trial and are trained
with binary cross-entropy with logits.

### Architectural comparison

The two models consume the same processed EEG but impose very different
inductive biases.

| Aspect | CBraMod | EEGSimpleConv |
|---|---|---|
| Initialization | Large-scale pretrained EEG checkpoint | Trained from scratch |
| Input view | `[B, 32, 4, 200]` one-second patches | `[B, 32, 800]` continuous sequence |
| Main operator | Criss-cross transformer attention | One-dimensional temporal convolutions |
| Spatial modeling | Explicit attention across channels at a fixed temporal patch | Implicit channel mixing through convolutional feature maps |
| Temporal modeling | Explicit attention across patches within each channel | Local temporal receptive fields enlarged by stacked convolutions and pooling |
| Positional information | Asymmetric conditional positional encoding | Encoded implicitly by convolution order and receptive fields |
| Receptive field | Global spatial and temporal interactions within each block | Primarily local, becoming wider with depth and pooling |
| Representation reuse | General EEG representation transferable across tasks | Task-specific representation |
| Expected cost | Higher parameter, memory, and latency cost | Smaller and operationally simpler |

#### CBraMod

CBraMod first divides each four-second, 200 Hz trial into four one-second
patches:

```text
shared input: [32 channels, 800 samples]
CBraMod view: [32 channels, 4 patches, 200 samples per patch]
```

Each patch is encoded by a small convolutional patch encoder. In the reference
architecture, the patch encoder contains three one-dimensional convolutional
layers with group normalization and GELU activations.

The resulting tokens form a two-dimensional channel-by-time grid. A standard
full-attention model would flatten this grid and model every patch relationship
with one attention operation. CBraMod instead uses a **criss-cross transformer**
that treats the two axes as structurally different:

- **spatial attention** relates different EEG channels at the same temporal
  patch;
- **temporal attention** relates different temporal patches within the same EEG
  channel.

The two attention paths operate in parallel and their outputs are combined in
each transformer block. This reflects the fact that correlation between scalp
locations and evolution through time are different kinds of dependencies rather
than interchangeable token relationships.

The reference backbone contains 12 criss-cross transformer blocks, a hidden
dimension of 200, a feed-forward dimension of 800, and eight attention heads
split between the spatial and temporal paths.

CBraMod also uses **asymmetric conditional positional encoding**. A depthwise
two-dimensional convolution derives positional information from the current
channel-by-time grid instead of relying only on a fixed learned table. This is
particularly useful for EEG because datasets may differ in channel count,
recording duration, and patch-grid dimensions.

The model was pretrained on a large EEG corpus using patch-based masked EEG
reconstruction. For SHU-MI, the released weights initialize the backbone, and
an all-patch classification head maps the complete set of downstream patch
representations to one binary logit.

#### EEGSimpleConv

EEGSimpleConv receives the same processed trial without converting it into
explicit channel-time tokens:

```text
EEGSimpleConv view: [32 channels, 800 samples]
```

The adapted model follows the reference architecture:

- internal temporal resampling;
- an initial temporal `Conv1d`;
- two convolutional blocks;
- batch normalization, ReLU, and max pooling;
- feature-width growth using `int(1.414 * width)`;
- global temporal average pooling;
- a binary classification head.

The model progressively learns local temporal filters and builds larger
receptive fields through stacked convolutions and pooling. Channel information
is mixed into learned feature maps, but the architecture does not explicitly
represent a channel-by-patch grid or apply separate spatial and temporal
attention.

Its main advantage is simplicity: it is trained end-to-end from scratch, has
fewer expensive global operations, and should require less memory and lower
inference latency.

### Why CBraMod can outperform EEGSimpleConv

CBraMod has several reasons to be expected to perform better on a limited-label,
cross-subject EEG task.

1. **Large-scale pretraining**

   The CBraMod backbone does not start from random parameters. Masked EEG
   pretraining exposes it to general temporal, spectral, and cross-channel
   regularities before SHU-MI fine-tuning. EEGSimpleConv must infer useful EEG
   features only from the 15 training subjects.

2. **Explicit separation of spatial and temporal dependencies**

   Motor-imagery information can depend both on how activity differs across
   scalp locations and on how patterns evolve over the four-second trial.
   Criss-cross attention models these axes separately. EEGSimpleConv can learn
   both indirectly, but its convolutional representation does not encode this
   distinction explicitly.

3. **Long-range interactions**

   Attention allows a temporal patch to interact directly with distant patches
   from the same channel and allows one electrode to interact directly with
   other electrodes at the same time. A convolutional model must build these
   relationships gradually through successive local layers and pooling.

4. **Adaptable positional representation**

   The conditional positional encoder is generated from the observed
   channel-time layout. This provides an architectural mechanism for adapting a
   pretrained model to downstream recordings with different dimensions,
   whereas the convolutional baseline learns only the structure present in the
   current task.

5. **Representation reuse and regularization**

   Pretraining constrains fine-tuning toward a representation already useful
   for EEG reconstruction across many recordings. This can act as a strong
   prior when the downstream dataset is comparatively small and may improve
   cross-subject generalization.

These are architectural hypotheses, not a substitute for measurement.
EEGSimpleConv may still match or outperform CBraMod if the motor-imagery
decision boundary is dominated by relatively simple local patterns, if the
pretraining corpus is not well aligned with SHU-MI, or if the larger model
overfits during fine-tuning.

### Fair comparison protocol

Both models use exactly the same:

- SHU-MI materialized examples;
- subject-level train, validation, and test splits;
- label encoding and amplitude scaling;
- metric implementations;
- five random seeds;
- validation-AUROC checkpoint-selection policy;
- final-test evaluation policy.

The storage backend may be HDF5 or Arrow. A parity test verifies that the two
backends contain identical SHU-MI signals, labels, subject/session metadata, and
split assignments.

The model-specific transformation is intentionally minimal:

```text
shared processed example: [32, 800]

CBraMod:
    [32, 800] → [32, 4, 200]

EEGSimpleConv:
    [32, 800] → [32, 800]
```

This ensures that differences are attributable to the architectures and their
documented optimization recipes rather than to different data.

CBraMod uses the released pretrained checkpoint and the fine-tuning settings
described in Task B. EEGSimpleConv uses its documented Adam optimizer and
epoch-40 step-decay schedule.

The following optional components from the broader EEGSimpleConv system are
excluded:

- Euclidean alignment;
- test-time batch-normalization adaptation;
- mixup;
- auxiliary subject classification.

Adding those components only to EEGSimpleConv would compare complete
task-specific pipelines rather than isolate the architectural trade-off.

### Objective and evaluation

For a batch of trials, each model returns one logit per example:

```text
logits shape: [batch]
```

Training uses binary cross-entropy with logits. The sigmoid of a logit is
interpreted as the probability of the encoded positive class, right-hand motor
imagery.

The reported predictive metrics are:

- **balanced accuracy**, which weights the left- and right-hand classes equally;
- **AUROC**, which measures ranking quality across decision thresholds;
- **AUC-PR**, which measures the precision-recall trade-off for the encoded
  positive class.

Validation AUROC selects the best checkpoint. The test subjects are evaluated
only after checkpoint selection.

### Measurements

The completed comparison reports:

- five-seed balanced accuracy, AUC-PR, and AUROC;
- per-seed matched comparisons;
- parameter count and trainable parameter count;
- checkpoint/state-dictionary size;
- validation-to-test performance gaps;
- checkpoint-selection epochs;
- fixed-threshold behavior and predicted-positive rates.

The repository also contains a hardware benchmarker for:

- batch-1 mean, median, and p95 latency;
- batch-64 throughput;
- peak GPU memory;
- training time when available.

These hardware measurements must be collected for both models on the same idle
device before making quantitative efficiency claims.

Run the complete workflow with:

```bash
make task-c
```

The generated comparison artifacts are stored in:

```text
reports/results_models_comparison/
├── aggregate_metrics.csv
├── comparison.json
├── comparison.md
├── model_complexity.json
├── per_seed_comparison.csv
├── performance_comparison.png
├── seedwise_auroc.png
└── validation_auroc_curves.png
```

### Results

The five-seed comparison is complete. Aggregate test performance is:

| Metric | CBraMod | EEGSimpleConv | CBraMod advantage | CBraMod seed wins |
|---|---:|---:|---:|---:|
| Balanced accuracy | **0.6149 ± 0.0148** | 0.5271 ± 0.0257 | +0.0878 | 5/5 |
| AUC-PR | **0.6845 ± 0.0133** | 0.5861 ± 0.0356 | +0.0984 | 5/5 |
| AUROC | **0.6742 ± 0.0235** | 0.6111 ± 0.0377 | +0.0631 | 5/5 |

Values are mean ± population standard deviation over the same five seeds.
CBraMod wins every metric for every matched seed: 15 wins out of 15
seed/metric comparisons.

Per-seed test results are:

| Seed | CB best epoch | CB BAcc | CB AUC-PR | CB AUROC | SC best epoch | SC BAcc | SC AUC-PR | SC AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 5 | 0.6233 | 0.6983 | 0.6948 | 50 | 0.5069 | 0.6379 | 0.6547 |
| 3408 | 26 | 0.5914 | 0.6635 | 0.6399 | 37 | 0.5450 | 0.5535 | 0.5860 |
| 3409 | 1 | 0.6354 | 0.6986 | 0.7031 | 25 | 0.4991 | 0.6192 | 0.6591 |
| 3410 | 3 | 0.6164 | 0.6851 | 0.6770 | 26 | 0.5680 | 0.5676 | 0.5849 |
| 3411 | 8 | 0.6081 | 0.6772 | 0.6564 | 26 | 0.5165 | 0.5522 | 0.5710 |

#### Model size

| Property | CBraMod | EEGSimpleConv |
|---|---:|---:|
| Parameters | 25,525,001 | 740,101 |
| State size | 97.37 MiB | 2.83 MiB |
| Initialization | Released pretrained backbone | Random initialization |

CBraMod is approximately 34.5 times larger. An important nuance is that the
selected `all_patch_reps` downstream classifier contains 20,641,201 parameters,
or approximately 80.9% of the complete CBraMod model. Therefore, much of the
size difference comes from the task-specific classification head rather than
from the pretrained backbone alone.

#### Stability and cross-subject generalization

CBraMod is more stable across seeds for all three headline metrics:

- balanced-accuracy standard deviation: 0.0148 versus 0.0257;
- AUC-PR standard deviation: 0.0133 versus 0.0356;
- AUROC standard deviation: 0.0235 versus 0.0377.

The mean validation-to-test AUROC drop is 0.0640 for CBraMod and 0.0873 for
EEGSimpleConv. The larger drop for EEGSimpleConv suggests greater sensitivity
to the identity and distribution of the held-out subjects.

CBraMod also adapts substantially earlier. Its median validation-selected epoch
is 5, compared with 26 for EEGSimpleConv. This is consistent with the
pretrained backbone already providing useful EEG features, while the
convolutional model must learn its representation from the 15 training
subjects.

#### Fixed-threshold behavior

EEGSimpleConv's AUROC and AUC-PR show that it learns useful ranking information,
but its balanced accuracy is often close to chance because its score offset is
unstable at the fixed 0.5 threshold.

Across seeds, its predicted-positive rate ranges from 1.8% to 98.7%. CBraMod's
range is much narrower, from 40.7% to 58.7%. Several EEGSimpleConv runs therefore
predict almost every test example as one class even when their AUROC remains
above 0.5.

This points to a calibration or cross-subject distribution-shift problem in
addition to weaker ranking performance. Batch-normalization statistics are a
plausible contributor. Selecting the decision threshold on the validation set
could improve balanced accuracy, but it would not improve AUROC or AUC-PR and
should be reported as a separate calibration experiment rather than replacing
the primary protocol retroactively.

#### Interpretation

The results support the practical advantage of CBraMod on this cross-subject
motor-imagery task. The most plausible contributors are:

1. **large-scale masked EEG pretraining**, which provides useful signal
   representations before SHU-MI fine-tuning;
2. **criss-cross spatial-temporal attention**, which explicitly separates
   relationships across electrodes from relationships across temporal patches;
3. **global interactions**, which allow distant patches or channels to
   communicate directly rather than only through stacked local convolutions;
4. **the spectral branch in the patch encoder**, which exposes frequency-domain
   information explicitly;
5. **stronger adaptation and stability**, reflected by earlier selected
   checkpoints, lower seed variance, and a smaller validation-to-test gap.

However, this experiment is a comparison of the intended practical systems,
not a pure causal architecture ablation. CBraMod is pretrained and uses its own
fine-tuning recipe, whereas EEGSimpleConv is trained from scratch with its
documented optimizer schedule. The measured gain therefore cannot be attributed
solely to attention versus convolution.

Useful follow-up ablations would include:

- CBraMod from random initialization;
- a frozen CBraMod backbone with a small classifier;
- an average-pooled CBraMod head to reduce the 20.6M-parameter downstream head;
- EEGSimpleConv with validation-set threshold calibration;
- EEGSimpleConv with Euclidean alignment, session normalization, mixup, and
  carefully controlled batch-normalization adaptation.

The complete evidence, figures, and machine-readable results are available in
[`results_models_comparison/comparison.md`](results_models_comparison/comparison.md)
and the accompanying JSON/CSV files.

### When to prefer each architecture

#### Prefer CBraMod when

- transfer from large-scale pretrained EEG representations matters;
- labelled downstream data is limited;
- reuse across several EEG datasets or tasks is expected;
- cross-subject or cross-dataset generalization is a priority;
- compute and memory budgets permit a transformer-style backbone;
- the learned representation may be reused beyond one classifier.

#### Prefer EEGSimpleConv when

- latency, memory use, and operational simplicity dominate;
- training from scratch on one task is sufficient;
- deployment targets constrained hardware;
- the convolutional baseline obtains comparable predictive performance;
- fast experimentation and maintainability matter more than representation
  reuse.

A comparable EEGSimpleConv result would be scientifically meaningful because it
would question whether CBraMod's additional complexity is justified for this
specific motor-imagery task. It would not invalidate CBraMod as a foundation
model, whose broader value includes transfer across heterogeneous EEG tasks and
datasets.

## Quality and reproducibility checks

The suite covers preprocessing, audits, models, checkpoint loading, metrics, training, multi-seed aggregation, benchmarks, comparison generation, Arrow/HDF5 parity, and MAT/EDF equivalence. The current integrated repository previously completed:

```text
26 tests passed
Ruff checks passed
Pyright reported 0 errors and 0 warnings
Model smoke test passed
```

Exact commands and test groups are documented in [`../DESIGN.md`](../DESIGN.md).
