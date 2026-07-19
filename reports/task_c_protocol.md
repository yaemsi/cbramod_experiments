# Task C protocol: EEGSimpleConv versus CBraMod

## Question

Can a straightforward convolutional model trained from scratch match the
pretrained CBraMod foundation model on the same cross-subject SHU-MI task, and
what efficiency trade-offs does each model make?

## Controlled factors

Both models use the same:

- 11,988 processed SHU-MI examples;
- subjects 1–15 / 16–20 / 21–25 split;
- signal scaling and labels;
- training batches and seed set;
- binary classification objective;
- validation-AUROC checkpoint selection;
- balanced accuracy, AUC-PR, and AUROC implementation;
- final test evaluation only after checkpoint selection.

## Model-specific factors

CBraMod uses its released pretrained checkpoint and paper-aligned AdamW/cosine
fine-tuning settings. EEGSimpleConv is initialized from scratch and uses its
documented 50-epoch Adam routine with a 10× learning-rate decay at epoch 40.
This gives each architecture a defensible training schedule while holding the
data and evaluation protocol fixed.

## Deliberately excluded EEGSimpleConv pipeline features

The original EEGSimpleConv work studies a broader pipeline containing Euclidean
alignment, session-wise standardization, recomputation of batch-normalization
statistics on held-out subjects, mixup, and subject-wise regularization. These
are excluded from the primary comparison because the take-home asks for an
architecture comparison and CBraMod is not given equivalent additions.

This decision should be reported explicitly. An optional follow-up experiment
could add these components as an "EEGSimpleConv full pipeline" variant, but it
is not necessary for the main deliverable.

## Required outputs

1. Five-seed predictive metrics with mean and standard deviation.
2. Per-seed metrics and selected epochs.
3. Parameter count and serialized state size.
4. Batch-1 latency for online/interactive deployment.
5. Batch-64 throughput and peak GPU memory for offline processing.
6. A discussion of accuracy, stability, training cost, inference cost, and
   deployment constraints.

## Interpretation

Prefer CBraMod when pretraining gives a meaningful improvement, labeled data are
scarce, or one model must transfer across many EEG tasks. Prefer EEGSimpleConv
when performance is comparable and low latency, operational simplicity, small
checkpoints, and easier debugging are more valuable.


## Completed result

The five-seed experiment is complete. CBraMod obtained balanced accuracy
`0.6149 ± 0.0148`, AUC-PR `0.6845 ± 0.0133`, and AUROC
`0.6742 ± 0.0235`. EEGSimpleConv obtained `0.5271 ± 0.0257`,
`0.5861 ± 0.0356`, and `0.6111 ± 0.0377`, respectively
(mean ± population SD).

CBraMod won every metric for every matched seed. EEGSimpleConv is approximately
34.5× smaller, but showed greater run-to-run variance and unstable calibration
at the fixed 0.5 decision threshold.

See [`task_c/comparison.md`](task_c/comparison.md) for the full analysis.
