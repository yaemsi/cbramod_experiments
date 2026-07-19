# EEGSimpleConv SHU-MI five-seed results

All five runs use the full 11,988-example SHU-MI dataset and the same
subject-level split and metric implementation as the CBraMod reproduction.

## Aggregate test results

| Metric | Mean ± population SD | Min | Max |
|---|---:|---:|---:|
| Balanced accuracy | 0.5271 ± 0.0257 | 0.4991 | 0.5680 |
| AUC-PR | 0.5861 ± 0.0356 | 0.5522 | 0.6379 |
| AUROC | 0.6111 ± 0.0377 | 0.5710 | 0.6591 |

## Per-seed results

| Seed | Best epoch | Balanced accuracy | AUC-PR | AUROC | Predicted positive rate |
|---:|---:|---:|---:|---:|---:|
| 3407 | 50 | 0.5069 | 0.6379 | 0.6547 | 98.7% |
| 3408 | 37 | 0.5450 | 0.5535 | 0.5860 | 81.6% |
| 3409 | 25 | 0.4991 | 0.6192 | 0.6591 | 1.8% |
| 3410 | 26 | 0.5680 | 0.5676 | 0.5849 | 59.4% |
| 3411 | 26 | 0.5165 | 0.5522 | 0.5710 | 92.3% |

## Main observation

The model ranks examples above chance, but the fixed 0.5 decision threshold is
unstable across seeds. This produces extreme class prediction rates and keeps
balanced accuracy close to chance in several runs. The issue should be
distinguished from ranking performance and investigated through calibration,
normalization, and cross-subject shift analyses.
