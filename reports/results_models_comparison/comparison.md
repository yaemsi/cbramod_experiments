# Task C — CBraMod versus EEGSimpleConv on SHU-MI

## Experimental question

This experiment compares the practical downstream choices specified by the
homework: fine-tuning the released pretrained CBraMod model versus training
EEGSimpleConv from scratch on the same SHU-MI examples. It is a controlled data
and evaluation comparison, but not a pure causal architecture ablation because
pretraining and optimizer recipes also differ.

## Controlled protocol

Both models use the same 11,988 examples, subject split (1–15 / 16–20 / 21–25),
signal scaling, labels, batch size, five seeds, binary loss, validation-AUROC
checkpoint selection, and test metric implementation.

## Test performance

Mean ± population standard deviation over five seeds:

| Metric | CBraMod | EEGSimpleConv | CBraMod advantage | CBraMod wins |
|---|---:|---:|---:|---:|
| Balanced accuracy | 0.6149 ± 0.0148 | 0.5271 ± 0.0257 | +0.0878 | 5/5 |
| AUC-PR | 0.6845 ± 0.0133 | 0.5861 ± 0.0356 | +0.0984 | 5/5 |
| AUROC | 0.6742 ± 0.0235 | 0.6111 ± 0.0377 | +0.0631 | 5/5 |

CBraMod wins on every metric for every matched seed. The largest average
differences are 0.0984 AUC-PR points and 0.0878 balanced-accuracy points.

## Per-seed results

| Seed | CB best epoch | CB BAcc | CB AUC-PR | CB AUROC | SC best epoch | SC BAcc | SC AUC-PR | SC AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 5 | 0.6233 | 0.6983 | 0.6948 | 50 | 0.5069 | 0.6379 | 0.6547 |
| 3408 | 26 | 0.5914 | 0.6635 | 0.6399 | 37 | 0.5450 | 0.5535 | 0.5860 |
| 3409 | 1 | 0.6354 | 0.6986 | 0.7031 | 25 | 0.4991 | 0.6192 | 0.6591 |
| 3410 | 3 | 0.6164 | 0.6851 | 0.6770 | 26 | 0.5680 | 0.5676 | 0.5849 |
| 3411 | 8 | 0.6081 | 0.6772 | 0.6564 | 26 | 0.5165 | 0.5522 | 0.5710 |

## Architecture and model size

| Property | CBraMod | EEGSimpleConv |
|---|---|---|
| Input view | 32 channels × four 200-sample patches | 32 channels × 800 samples, internally resampled to 80 Hz |
| Core operation | 12 criss-cross attention/FFN layers | temporal Conv1d blocks + max pooling |
| Cross-channel modeling | explicit spatial attention across electrodes | learned mixing in convolutional feature maps |
| Long-range temporal modeling | attention across patches | local convolutional features followed by global average pooling |
| Spectral representation | explicit rFFT branch in patch embedding | learned indirectly from waveforms |
| Initialization | released pretrained backbone | random |
| Parameters | 25,525,001 | 740,101 |
| State size | 97.37 MiB | 2.83 MiB |

CBraMod is 34.5× larger.
The chosen `all_patch_reps` downstream head alone contains
20,641,201 parameters
(80.9% of the full model), so much
of the size difference comes from the task-specific classifier rather than the
foundation backbone alone.

## Stability and generalization

CBraMod has lower run-to-run dispersion on all three headline metrics.
EEGSimpleConv's population SD is 1.7× larger for balanced accuracy,
2.7× for AUC-PR, and
1.6× for AUROC.

The mean validation-to-test AUROC drop is
0.0640 for CBraMod and
0.0873 for
EEGSimpleConv. The larger SimpleConv gap suggests greater sensitivity to the
identity and distribution of the held-out subjects.

CBraMod reaches its validation-selected checkpoint much earlier (median epoch
5) than EEGSimpleConv (median epoch 26). This is consistent with useful
pretrained representations: CBraMod adapts quickly, while the convolutional
baseline must learn the task representation from scratch.

## Fixed-threshold behavior

EEGSimpleConv's AUROC and AUC-PR remain above chance, but its balanced accuracy
is often close to 0.5 because the score offset is unstable at the fixed 0.5
threshold:

| Seed | Predicted positive rate | Specificity | Sensitivity |
|---:|---:|---:|---:|
| 3407 | 98.7% | 0.020 | 0.994 |
| 3408 | 81.6% | 0.229 | 0.861 |
| 3409 | 1.8% | 0.981 | 0.017 |
| 3410 | 59.4% | 0.474 | 0.662 |
| 3411 | 92.3% | 0.093 | 0.940 |

The predicted-positive rate ranges from
1.8% to 98.7% across
SimpleConv seeds, compared with 40.7% to
58.7% for CBraMod. This indicates a calibration or
domain-shift problem in addition to weaker ranking quality. Batch-normalization
statistics are a plausible contributor under cross-subject shift.

Selecting a threshold on the validation set could improve SimpleConv balanced
accuracy, but would not change AUROC/AUC-PR and should be reported as a separate
calibration experiment rather than retroactively replacing the primary
paper-aligned protocol.

## Interpretation

On this SHU-MI cross-subject task, EEGSimpleConv does learn useful signal
(AUROC 0.6111), but it does not
match CBraMod. The pretrained model has a clear performance and stability
advantage, winning all 15 matched seed/metric comparisons.

EEGSimpleConv remains attractive operationally: it has only
740,101 parameters and a
2.83 MiB state, making it easier
to deploy, inspect, and retrain. However, same-device latency, throughput, peak
memory, and wall-clock measurements were not included in the supplied run
artifacts, so those advantages should be described as architectural
expectations until `make benchmark-models` is run on the same idle GPU.

## When to prefer each model

Prefer CBraMod when cross-subject accuracy, stability, label efficiency, or
reuse across multiple EEG tasks matters and the additional model size is
acceptable.

Prefer EEGSimpleConv when memory, implementation simplicity, or edge deployment
dominates and lower predictive performance is acceptable. Before deployment,
its threshold calibration and batch-normalization behavior should be addressed.

## Fairness caveat and useful follow-up ablations

The practical comparison intentionally uses CBraMod pretrained and
EEGSimpleConv from scratch, matching their intended use. It therefore cannot
attribute the whole gain to attention versus convolution alone.

Useful optional ablations are:

1. CBraMod from random initialization;
2. frozen CBraMod backbone plus a small head;
3. CBraMod with average-pooling classifier to reduce the 20.6M-parameter head;
4. EEGSimpleConv with validation-set threshold calibration;
5. EEGSimpleConv's broader recipe: Euclidean alignment, session normalization,
   mixup, and carefully defined BN adaptation.
