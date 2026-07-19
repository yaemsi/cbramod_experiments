# CBraMod SHU-MI Five-Seed Reproduction

## Result summary

| Metric | Reproduction, mean ± SD | Paper, mean ± SD | Difference |
|---|---:|---:|---:|
| Balanced accuracy | 0.6149 ± 0.0148 | 0.6370 ± 0.0151 | -0.0221 |
| AUC-PR | 0.6845 ± 0.0133 | 0.7139 ± 0.0088 | -0.0294 |
| AUROC | 0.6742 ± 0.0235 | 0.6988 ± 0.0068 | -0.0246 |

The main table uses population standard deviation (`ddof=0`), which is common in machine-learning experiment summaries. The corresponding sample standard deviations are available in `shu_mi_5seed_summary.csv`.

## Per-seed test results

| Seed | Best epoch | Balanced accuracy | AUC-PR | AUROC |
|---:|---:|---:|---:|---:|
| 3407 | 5 | 0.6233 | 0.6983 | 0.6948 |
| 3408 | 26 | 0.5914 | 0.6635 | 0.6399 |
| 3409 | 1 | 0.6354 | 0.6986 | 0.7031 |
| 3410 | 3 | 0.6164 | 0.6851 | 0.6770 |
| 3411 | 8 | 0.6081 | 0.6772 | 0.6564 |

## Interpretation

The reproduction is in the same general performance range as the paper, but it is not an exact numerical match. Mean scores are lower by 0.0221 balanced-accuracy points, 0.0294 AUC-PR points, and 0.0246 AUROC points.

Seed 3409 is the closest overall run, reaching 0.6354 balanced accuracy and 0.7031 AUROC. Seed 3408 is a clear low outlier, especially for AUROC (0.6399), and contributes substantially to the larger run-to-run variance.

The validation-to-test drop is meaningful, suggesting sensitivity to the held-out subject group rather than a preprocessing failure. The best checkpoint occurs at a median of epoch 5, while training continues to epoch 50 and training loss falls near zero. This indicates strong overfitting after the early epochs, although selecting by validation AUROC protects the reported test result.

## Reproducibility checks

All five runs used the same dataset protocol, pretrained-checkpoint hash, model configuration, and training hyperparameters; only the random seed and output directory differed. The run metadata reports 11,988 examples with subjects 1–15 for training, 16–20 for validation, and 21–25 for testing.

## Recommended wording for the submission

> We reproduced CBraMod on SHU-MI over five random seeds using the paper's subject-level split and released pretrained checkpoint. Our implementation obtained balanced accuracy of 0.6149 ± 0.0148, AUC-PR of 0.6845 ± 0.0133, and AUROC of 0.6742 ± 0.0235. These results are in the same range as the reported values, though our mean scores were approximately 2–3 percentage points lower and showed greater seed sensitivity, particularly for AUROC. We observed rapid overfitting, with validation-selected checkpoints typically occurring during the first several epochs.
