# CBraMod homework development and reproduction commands.
# Examples:
#   make sync
#   make preprocess RAW_DIR=/path/to/shu/mat_files
#   make inspect-data
#   make check-checkpoint
#   make reproduce-cbramod

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python
PACKAGE := cbramod_experiments
TEST_DIR := tests

SHU_ROOT ?= resources/data/shu-mi_dataset
RAW_DIR ?= $(SHU_ROOT)/mat_files
DATASET ?= $(SHU_ROOT)/preprocessed/shu_mi.h5
DATA_BACKEND ?= hdf5
HARMONIZED_SHU_DIR ?= resources/data/harmonized/shu_mi
HARMONIZED_SHU_MANIFEST ?= $(HARMONIZED_SHU_DIR)/manifest.parquet
SHU_EDF_DIR ?= $(SHU_ROOT)/edf_files
SHU_EVENTS_DIR ?= $(SHU_ROOT)/events
HARMONIZED_SHU_EDF_DIR ?= resources/data/harmonized/shu_mi_edf
HBN_ROOT ?= resources/data/hbn
HARMONIZED_HBN_DIR ?= resources/data/harmonized/hbn_subset
HARMONIZED_HBN_MANIFEST ?= $(HARMONIZED_HBN_DIR)/manifest.parquet
HBN_DATASET_ID ?= hbn
HBN_TARGET_RATE ?= 200
HBN_WINDOW_SECONDS ?= 4
HBN_STRIDE_SECONDS ?= 4
HBN_LIMIT_RECORDINGS ?= 3
ARROW_RECORDS_PER_BATCH ?= 256
ARROW_BATCHES_PER_SHARD ?= 16
CBRAMOD_CONFIG ?= configs/cbramod.yaml
SIMPLECONV_CONFIG ?= configs/eegsimpleconv.yaml
CHECKPOINT ?=
OUTPUT_DIR ?= outputs/cbramod_shu_mi
SIMPLECONV_OUTPUT_DIR ?= outputs/eegsimpleconv_shu_mi
TASK_C_OUTPUT_DIR ?= reports/results_models_comparison
CBRAMOD_SUMMARY ?= $(OUTPUT_DIR)/summary.json
SIMPLECONV_SUMMARY ?= $(SIMPLECONV_OUTPUT_DIR)/summary.json
CBRAMOD_BENCHMARK ?= $(TASK_C_OUTPUT_DIR)/cbramod_benchmark.json
SIMPLECONV_BENCHMARK ?= $(TASK_C_OUTPUT_DIR)/eegsimpleconv_benchmark.json
REPRO_SEEDS ?= 3407 3408 3409 3410 3411
BENCHMARK_DEVICE ?= auto
BENCHMARK_BATCHES ?= 1 64
BENCHMARK_WARMUP ?= 20
BENCHMARK_ITERATIONS ?= 100
DATA_NOTEBOOK ?= notebooks/shu_mi_data_exploration.ipynb
SAMPLE_ROOT ?= data/fixtures/shu_mi_single_session
SAMPLE_STEM ?= sub-001_ses-01_task_motorimagery
PYTEST_ARGS ?=
OVERWRITE ?= 0
STRICT ?= 1

ifeq ($(OVERWRITE),1)
PREPROCESS_FLAGS := --overwrite
else
PREPROCESS_FLAGS :=
endif

ifeq ($(STRICT),1)
STRICT_DATA_FLAG := --strict-data
INSPECT_STRICT_FLAG := --strict
else
STRICT_DATA_FLAG :=
INSPECT_STRICT_FLAG :=
endif

ifneq ($(strip $(CHECKPOINT)),)
CHECKPOINT_FLAG := --checkpoint-path "$(CHECKPOINT)"
else
CHECKPOINT_FLAG :=
endif

.PHONY: help all sync install lock update smoke main test test-all test-integration test-verbose \
        lint lint-fix format format-check typecheck check ci hooks \
        preprocess inspect-data harmonize-shu harmonize-shu-edf harmonize-hbn \
        inspect-harmonized compare-backends check-checkpoint train-cbramod train-simpleconv \
        reproduce-cbramod reproduce-cbramod-debug reproduce-simpleconv \
        benchmark-cbramod benchmark-simpleconv benchmark-models compare-models task-c \
        stage-sample sample-preprocess sample-inspect sample-harmonize sample-harmonize-edf \
        sample-compare-backends sample-harmonize-bids explore-data render-data-notebook \
        clean clean-data distclean

all: check ## Run the full local validation suite.

help: ## Show available targets.
	@printf "\nUsage: make <target> [VARIABLE=value]\n\n"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nCommon variables:\n"
	@printf "  RAW_DIR=%s\n  DATASET=%s  DATA_BACKEND=%s\n" "$(RAW_DIR)" "$(DATASET)" "$(DATA_BACKEND)"
	@printf "  HARMONIZED_SHU_MANIFEST=%s\n  HBN_ROOT=%s\n" "$(HARMONIZED_SHU_MANIFEST)" "$(HBN_ROOT)"
	@printf "  CHECKPOINT=<local .pth>\n"
	@printf "  OUTPUT_DIR=%s\n  SIMPLECONV_OUTPUT_DIR=%s\n" "$(OUTPUT_DIR)" "$(SIMPLECONV_OUTPUT_DIR)"
	@printf "  REPRO_SEEDS='%s'  BENCHMARK_DEVICE=%s\n" "$(REPRO_SEEDS)" "$(BENCHMARK_DEVICE)"
	@printf "  OVERWRITE=0|1  STRICT=0|1  PYTEST_ARGS='...'\n\n"

sync: ## Create/update the uv environment and install the project.
	$(UV) sync

install: sync ## Alias for sync.

lock: ## Regenerate uv.lock without upgrading declared dependencies.
	$(UV) lock

update: ## Upgrade dependencies allowed by pyproject.toml.
	$(UV) lock --upgrade
	$(UV) sync

smoke: ## Run a quick model and metrics smoke test.
	$(UV) run $(PYTHON) -m main smoke

main: smoke ## Backward-compatible alias for smoke.

test: ## Run fast tests that do not require real local EEG files.
	$(UV) run $(PYTHON) -m pytest $(TEST_DIR) -q -m "not integration" $(PYTEST_ARGS)

test-integration: ## Run tests requiring real SHU-MI files/full manifests.
	SHU_MI_ROOT="$(SHU_ROOT)" $(UV) run $(PYTHON) -m pytest $(TEST_DIR) -q -m integration $(PYTEST_ARGS)

test-all: ## Run unit and integration tests together.
	SHU_MI_ROOT="$(SHU_ROOT)" $(UV) run $(PYTHON) -m pytest $(TEST_DIR) -q $(PYTEST_ARGS)

test-verbose: ## Run fast tests with verbose output.
	$(UV) run $(PYTHON) -m pytest $(TEST_DIR) -vv -m "not integration" $(PYTEST_ARGS)

lint: ## Check lint rules without modifying files.
	$(UV) run ruff check $(PACKAGE) $(TEST_DIR) main.py

lint-fix: ## Apply Ruff safe automatic fixes.
	$(UV) run ruff check $(PACKAGE) $(TEST_DIR) main.py --fix

format: ## Format source and tests.
	$(UV) run ruff format $(PACKAGE) $(TEST_DIR) main.py

format-check: ## Verify formatting without changing files.
	$(UV) run ruff format --check $(PACKAGE) $(TEST_DIR) main.py

typecheck: ## Run static type checking.
	$(UV) run pyright $(PACKAGE) main.py

check: format-check lint typecheck test smoke ## Run all local quality checks.

ci: check ## CI-friendly validation alias.

hooks: ## Install pre-commit hooks.
	$(UV) run pre-commit install
	@echo "Pre-commit hooks installed."

preprocess: ## Preprocess the full SHU-MI MATLAB archive into HDF5.
	$(UV) run $(PYTHON) -m main preprocess --raw-dir "$(RAW_DIR)" --output "$(DATASET)" $(PREPROCESS_FLAGS)

inspect-data: ## Validate shapes, classes, subject splits, leakage, and paper sample count.
	$(UV) run $(PYTHON) -m main inspect-data --data "$(DATASET)" $(INSPECT_STRICT_FLAG)

harmonize-shu: ## Convert full SHU-MI MAT data into Parquet manifest + Arrow shards.
	$(UV) run $(PYTHON) -m main harmonize-shu --source mat --raw-dir "$(RAW_DIR)" --output-dir "$(HARMONIZED_SHU_DIR)" --records-per-batch $(ARROW_RECORDS_PER_BATCH) --batches-per-shard $(ARROW_BATCHES_PER_SHARD) $(PREPROCESS_FLAGS)

harmonize-shu-edf: ## Reconstruct SHU-MI trials from EDF/events into the same Arrow schema.
	$(UV) run $(PYTHON) -m main harmonize-shu --source edf --raw-dir "$(SHU_EDF_DIR)" --events-root "$(SHU_EVENTS_DIR)" --output-dir "$(HARMONIZED_SHU_EDF_DIR)" --records-per-batch $(ARROW_RECORDS_PER_BATCH) --batches-per-shard $(ARROW_BATCHES_PER_SHARD) $(PREPROCESS_FLAGS)

harmonize-hbn: ## Harmonize a small HBN BIDS EDF/BDF/SET subset into the shared schema.
	$(UV) run $(PYTHON) -m main harmonize-bids --root "$(HBN_ROOT)" --output-dir "$(HARMONIZED_HBN_DIR)" --dataset-id "$(HBN_DATASET_ID)" --target-sampling-rate $(HBN_TARGET_RATE) --window-seconds $(HBN_WINDOW_SECONDS) --stride-seconds $(HBN_STRIDE_SECONDS) --limit-recordings $(HBN_LIMIT_RECORDINGS) --records-per-batch $(ARROW_RECORDS_PER_BATCH) --batches-per-shard $(ARROW_BATCHES_PER_SHARD) $(PREPROCESS_FLAGS)

inspect-harmonized: ## Summarize a harmonized manifest; STRICT=1 also validates full SHU protocol.
	$(UV) run $(PYTHON) -m main inspect-harmonized --manifest "$(HARMONIZED_SHU_MANIFEST)" $(if $(filter 1,$(STRICT)),--strict-shu,)

compare-backends: ## Verify HDF5 and Arrow contain numerically identical SHU samples.
	$(UV) run $(PYTHON) -m main compare-backends --hdf5 "$(DATASET)" --manifest "$(HARMONIZED_SHU_MANIFEST)"

check-checkpoint: ## Download or validate the released CBraMod checkpoint and its SHA256.
	$(UV) run $(PYTHON) -m main check-checkpoint --config "$(CBRAMOD_CONFIG)" $(CHECKPOINT_FLAG)

train-cbramod: ## Run one CBraMod seed with the paper-aligned configuration.
	$(UV) run $(PYTHON) -m main train --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --data-backend "$(DATA_BACKEND)" --output-dir "$(OUTPUT_DIR)/seed_3407" $(CHECKPOINT_FLAG) $(STRICT_DATA_FLAG)

train-simpleconv: ## Run one EEGSimpleConv seed on the same processed data.
	$(UV) run $(PYTHON) -m main train --config "$(SIMPLECONV_CONFIG)" --data "$(DATASET)" --data-backend "$(DATA_BACKEND)" --output-dir "$(SIMPLECONV_OUTPUT_DIR)/seed_3407" $(STRICT_DATA_FLAG)

reproduce-cbramod: ## Run five seeds and write mean/std metrics to summary.json.
	$(UV) run $(PYTHON) -m main reproduce --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --data-backend "$(DATA_BACKEND)" --output-dir "$(OUTPUT_DIR)" --seeds $(REPRO_SEEDS) $(CHECKPOINT_FLAG)

reproduce-simpleconv: ## Run five EEGSimpleConv seeds and aggregate the metrics.
	$(UV) run $(PYTHON) -m main reproduce --config "$(SIMPLECONV_CONFIG)" --data "$(DATASET)" --data-backend "$(DATA_BACKEND)" --output-dir "$(SIMPLECONV_OUTPUT_DIR)" --seeds $(REPRO_SEEDS)

benchmark-cbramod: ## Benchmark CBraMod architecture without downloading weights.
	$(UV) run $(PYTHON) -m main benchmark --config "$(CBRAMOD_CONFIG)" --output "$(CBRAMOD_BENCHMARK)" --device "$(BENCHMARK_DEVICE)" --batch-sizes $(BENCHMARK_BATCHES) --warmup $(BENCHMARK_WARMUP) --iterations $(BENCHMARK_ITERATIONS) --random-init

benchmark-simpleconv: ## Benchmark EEGSimpleConv on the same device and input shape.
	$(UV) run $(PYTHON) -m main benchmark --config "$(SIMPLECONV_CONFIG)" --output "$(SIMPLECONV_BENCHMARK)" --device "$(BENCHMARK_DEVICE)" --batch-sizes $(BENCHMARK_BATCHES) --warmup $(BENCHMARK_WARMUP) --iterations $(BENCHMARK_ITERATIONS) --random-init

benchmark-models: benchmark-cbramod benchmark-simpleconv ## Benchmark both architectures.

compare-models: ## Generate Task C comparison.json and comparison.md.
	$(UV) run $(PYTHON) -m main compare --cbramod-summary "$(CBRAMOD_SUMMARY)" --simpleconv-summary "$(SIMPLECONV_SUMMARY)" --cbramod-benchmark "$(CBRAMOD_BENCHMARK)" --simpleconv-benchmark "$(SIMPLECONV_BENCHMARK)" --output-dir "$(TASK_C_OUTPUT_DIR)"

task-c: reproduce-simpleconv benchmark-models compare-models ## Run the complete EEGSimpleConv comparison workflow.

reproduce-cbramod-debug: ## Run the reproduction command without requiring all 25 subjects.
	$(UV) run $(PYTHON) -m main reproduce --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --data-backend "$(DATA_BACKEND)" --output-dir "$(OUTPUT_DIR)" --seeds $(REPRO_SEEDS) $(CHECKPOINT_FLAG) --allow-incomplete-data

stage-sample: ## Stage one subject/session from the full local SHU-MI archive.
	rm -rf "$(SAMPLE_ROOT)"
	mkdir -p "$(SAMPLE_ROOT)/mat_files" "$(SAMPLE_ROOT)/edf_files" "$(SAMPLE_ROOT)/events"
	@mat=$$(find "$(RAW_DIR)" -type f -name "$(SAMPLE_STEM)_eeg.mat" -print -quit); \
	test -n "$$mat" || { echo "Missing $(SAMPLE_STEM)_eeg.mat below $(RAW_DIR)"; exit 1; }; \
	cp "$$mat" "$(SAMPLE_ROOT)/mat_files/"
	@edf=$$(find "$(SHU_EDF_DIR)" -type f -name "$(SAMPLE_STEM)_eeg.edf" -print -quit); \
	test -n "$$edf" || { echo "Missing $(SAMPLE_STEM)_eeg.edf below $(SHU_EDF_DIR)"; exit 1; }; \
	cp "$$edf" "$(SAMPLE_ROOT)/edf_files/"
	@events=$$(find "$(SHU_EVENTS_DIR)" -type f -name "$(SAMPLE_STEM)_events.tsv" -print -quit); \
	test -n "$$events" || { echo "Missing $(SAMPLE_STEM)_events.tsv below $(SHU_EVENTS_DIR)"; exit 1; }; \
	cp "$$events" "$(SAMPLE_ROOT)/events/"


sample-preprocess: stage-sample ## Preprocess one staged 100-trial SHU-MI session.
	$(UV) run $(PYTHON) -m main preprocess --raw-dir "$(SAMPLE_ROOT)/mat_files" --output data/processed/shu_mi_sample.h5 --overwrite

sample-inspect: ## Inspect the staged incomplete sample (warnings are expected).
	$(UV) run $(PYTHON) -m main inspect-data --data data/processed/shu_mi_sample.h5

sample-harmonize: stage-sample ## Build Arrow shards from one staged SHU-MI MAT session.
	$(UV) run $(PYTHON) -m main harmonize-shu --source mat --raw-dir "$(SAMPLE_ROOT)/mat_files" --output-dir data/harmonized/shu_mi_sample --records-per-batch 32 --batches-per-shard 2 --overwrite

sample-harmonize-edf: stage-sample ## Reconstruct the same staged session through EDF + events.
	$(UV) run $(PYTHON) -m main harmonize-shu --source edf --raw-dir "$(SAMPLE_ROOT)/edf_files" --events-root "$(SAMPLE_ROOT)/events" --output-dir data/harmonized/shu_mi_edf_sample --records-per-batch 32 --batches-per-shard 2 --overwrite

sample-compare-backends: sample-preprocess sample-harmonize ## Verify HDF5/Arrow parity on one staged session.
	$(UV) run $(PYTHON) -m main compare-backends --hdf5 data/processed/shu_mi_sample.h5 --manifest data/harmonized/shu_mi_sample/manifest.parquet

sample-harmonize-bids: stage-sample ## Exercise the generic BIDS reader on one staged EDF recording.
	$(UV) run $(PYTHON) -m main harmonize-bids --root "$(SAMPLE_ROOT)/edf_files" --output-dir data/harmonized/bids_sample --dataset-id bids-poc --target-sampling-rate 200 --window-seconds 4 --stride-seconds 4 --limit-recordings 1 --records-per-batch 32 --batches-per-shard 2 --overwrite

explore-data: ## Open the SHU-MI exploration notebook in JupyterLab.
	$(UV) run jupyter lab "$(DATA_NOTEBOOK)"

render-data-notebook: ## Execute the exploration notebook and export an HTML preview.
	$(UV) run jupyter nbconvert --to html --execute "$(DATA_NOTEBOOK)" --ExecutePreprocessor.timeout=300 --output-dir notebooks

clean: ## Remove caches and generated experiment outputs.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf build dist outputs

clean-data: ## Remove generated HDF5/Arrow data; raw data is preserved.
	rm -rf "$(SHU_ROOT)/preprocessed" \
	       "$(HARMONIZED_SHU_DIR)" \
	       "$(HARMONIZED_SHU_EDF_DIR)" \
	       "$(HARMONIZED_HBN_DIR)" \
	       data/processed data/harmonized data/fixtures

distclean: clean ## Remove caches plus the local virtual environment.
	rm -rf .venv
