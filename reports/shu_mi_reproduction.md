# SHU-MI CBraMod reproduction protocol

## Target result

The paper reports the following mean ± standard deviation over five runs:

| Metric | Paper |
|---|---:|
| Balanced accuracy | 0.6370 ± 0.0151 |
| AUC-PR | 0.7139 ± 0.0088 |
| AUROC | 0.6988 ± 0.0068 |

The paper describes 11,988 four-second examples, 32 channels, resampling from
250 Hz to 200 Hz, and the subject split 1–15 / 16–20 / 21–25.

## Observed five-seed reproduction

| Metric | Reproduction | Paper | Difference |
|---|---:|---:|---:|
| Balanced accuracy | 0.6149 ± 0.0148 | 0.6370 ± 0.0151 | -0.0221 |
| AUC-PR | 0.6845 ± 0.0133 | 0.7139 ± 0.0088 | -0.0294 |
| AUROC | 0.6742 ± 0.0235 | 0.6988 ± 0.0068 | -0.0246 |

The result is an approximate rather than exact reproduction. Seed 3409 reached
0.7031 AUROC, while seed 3408 reached 0.6399. This wide range and the consistent
validation-to-test drop indicate substantial seed and held-out-subject
sensitivity.

## Paper-aligned implementation choices

- Input scaling: divide EEG samples by 100 at load time.
- Patch representation: reshape `[B, 32, 800]` to `[B, 32, 4, 200]`.
- Released pretrained checkpoint and three-layer `all_patch_reps` classifier.
- 50 epochs, batch size 64, AdamW, weight decay 0.05.
- Backbone learning rate: `1e-4`.
- Classifier learning rate: `0.001 * sqrt(64 / 256) = 5e-4`.
- Cosine annealing after every optimizer step, minimum learning rate `1e-6`.
- Gradient norm clipping at 1.0.
- Best checkpoint selected by validation AUROC.
- AUC-PR computed by trapezoidal integration of the precision-recall curve,
  matching the released evaluator. Average precision is reported separately.

## Explicit deviations and clarifications

- The paper reports five-run statistics but does not identify the exact seeds.
  This repository uses `3407, 3408, 3409, 3410, 3411` and records them in the
  aggregate result.
- The released code evaluates the test split every time validation AUROC
  improves. This implementation evaluates test only after model selection to
  avoid repeated test-set access; the final selected checkpoint is unchanged.
- The released code stores pickled samples in LMDB. This implementation stores
  an inspectable HDF5 schema but applies the same resampling, label mapping,
  patching, and amplitude scaling.
- Full precision is the default because the released fine-tuning loop does not
  use autocast. AMP remains available as a configurable engineering option.

## Required validation before reporting results

`make inspect-data` must report:

- `paper_ready=True`;
- 11,988 examples;
- 32 channels and 800 points;
- subjects 1–15 only in train, 16–20 only in validation, 21–25 only in test;
- both classes present in every split;
- no overlapping or unassigned examples.

`make check-checkpoint` must report the released checkpoint SHA256:

`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`

## Run commands

```bash
make preprocess RAW_DIR=/absolute/path/to/mat_files OVERWRITE=1
make inspect-data
make check-checkpoint
make train-cbramod
make reproduce-cbramod
```

With a manually downloaded checkpoint:

```bash
make check-checkpoint CHECKPOINT=/absolute/path/pretrained_weights.pth
make reproduce-cbramod CHECKPOINT=/absolute/path/pretrained_weights.pth
```

The aggregate report is written to:

```text
outputs/cbramod_shu_mi/summary.json
```

Each seed directory contains the resolved configuration, runtime and dataset
manifest, training history, best checkpoint, validation metrics, and final test
metrics.
