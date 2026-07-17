# CBraMod vs EEGSimpleConv on SHU-MI

A compact, reproducible implementation for the CBraMod take-home assessment. The project
uses one shared SHU-MI data pipeline and evaluation protocol for both architectures.

## Current status

| Component | Status |
|---|---|
| Reproducible environment | Implemented; lock file must be regenerated on the target machine |
| SHU-MI subject split | Implemented and unit-tested |
| MATLAB to chunked HDF5 preprocessing | Implemented and validated on the included real SHU-MI sample |
| Worker-safe HDF5 data loader | Implemented |
| CBraMod architecture and Hugging Face checkpoint loader | Implemented; real checkpoint download not tested in this offline environment |
| EEGSimpleConv | Implemented |
| Shared training/evaluation loop | Implemented |
| Balanced accuracy, AUPRC and AUROC | Implemented and unit-tested |
| Real-data reproduction | Pending SHU-MI extraction |
| Multi-seed comparison | Pending real-data runs |

## Why HDF5 instead of reproducing the original LMDB pipeline exactly?

The original repository writes one pickled object per transaction and stores the split in a
special key. For this assessment, HDF5 makes the schema and metadata inspectable, supports
chunked/compressed arrays, and can be opened lazily inside each data-loader worker. The
preprocessing output stores signals, labels, subject IDs, session IDs, trial IDs, source
filenames, split indices, preprocessing parameters, and a schema version.

This is an engineering improvement, not a claim that HDF5 is universally preferable. For
terabyte-scale pretraining, sharded formats such as WebDataset, Zarr, or large binary/Parquet
shards would be evaluated based on storage and training-cluster benchmarks.

## Setup

Python 3.10 or 3.11 is recommended. Python 3.13 was removed because several scientific and
GPU packages may lag the newest interpreter.

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
```

Alternatively:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run tests and the synthetic smoke test:

```bash
python -m pytest
python -m main smoke
```

## Data preparation

Download and extract SHU-MI. The expected MATLAB variables, following the authors' code, are:

- `data`: `[trials, channels, time]`
- `labels`: binary labels encoded as either `1/2` or `0/1`

Inspect filenames before preprocessing:

```bash
find /path/to/shu -type f -name '*.mat' | sort | head -30
```

Then preprocess:

```bash
python -m cbramod_experiments.cli preprocess \
  --raw-dir /path/to/shu/mat \
  --output data/processed/shu_mi.h5
```

The preprocessor explicitly assigns:

- subjects 1-15 to train;
- subjects 16-20 to validation;
- subjects 21-25 to test.

It validates the expected four-second, 250 Hz input and resamples each trial to 800 samples
at 200 Hz, matching the public CBraMod SHU preprocessing.

The included real sample uses the filename
`sub-001_ses-01_task_motorimagery_eeg.mat`; the parser has been validated against this format.
The file contains 100 trials with shape `[100, 32, 1000]` and balanced labels encoded as `1/2`.
The preprocessing command converts it to `[100, 32, 800]` and maps the labels to `0/1`.
Before processing the full archive, the command still validates every file rather than assuming
that all filenames and shapes are correct.

## Training

CBraMod:

```bash
python -m cbramod_experiments.cli train --config configs/cbramod.yaml
```

EEGSimpleConv:

```bash
python -m cbramod_experiments.cli train --config configs/eegsimpleconv.yaml
```

Each run saves:

- `best_model.pt` selected by validation AUROC;
- `metrics.json` with validation and test metrics;
- `history.json` with the per-epoch trace.

The test split is evaluated only after model selection. Validation and test loaders are not
shuffled.

## Fair-comparison protocol

Both models use the same:

- resampled signals and explicit subject split;
- amplitude scaling;
- binary label mapping;
- balanced accuracy, AUPRC and AUROC implementation;
- validation-AUROC checkpoint selection;
- test evaluation code;
- random seeds for the final multi-seed experiment.

Model-specific optimization settings are kept in separate configuration files because the
architectures have different training dynamics. Both shared-data and shared-optimization
comparisons can be reported if compute permits.

## Repository layout

```text
cbramod_experiments/
├── cli.py
├── config.py
├── data/shu.py
├── models/cbramod.py
├── models/eegsimpleconv.py
├── train.py
├── evaluate.py
├── metrics.py
└── utils.py
configs/
tests/
answers.md
```

## Known limitations before real-data validation

1. Only one real SHU-MI file is included; full subject/session coverage still needs validation.
2. The official Hugging Face checkpoint download and key remapping must be tested online.
3. The paper-level hyperparameters and number of seeds must be confirmed against the
   camera-ready paper and repository.
4. HDF5 read throughput should be benchmarked with the full dataset and worker count.
5. Results cannot be reported until full-data training is completed.
