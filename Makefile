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

RAW_DIR ?= resources/data/shu-mi_dataset/mat_files
DATASET ?= resources/data/shu-mi_dataset/preprocessed/shu_mi.h5
CBRAMOD_CONFIG ?= configs/cbramod.yaml
SIMPLECONV_CONFIG ?= configs/eegsimpleconv.yaml
CHECKPOINT ?=
OUTPUT_DIR ?= outputs/cbramod_shu_mi
SIMPLECONV_OUTPUT_DIR ?= outputs/eegsimpleconv_shu_mi
TASK_C_OUTPUT_DIR ?= reports/task_c
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

.PHONY: help all sync install lock update smoke main test test-verbose \
        lint lint-fix format format-check typecheck check ci hooks hooks-run hooks-run-push hooks-uninstall \
        preprocess inspect-data check-checkpoint train-cbramod train-simpleconv \
        reproduce-cbramod reproduce-cbramod-debug reproduce-simpleconv \
        benchmark-cbramod benchmark-simpleconv benchmark-models compare-models task-c \
        sample-preprocess sample-inspect explore-data render-data-notebook clean clean-data distclean

all: check ## Run the full local validation suite.

help: ## Show available targets.
	@printf "\nUsage: make <target> [VARIABLE=value]\n\n"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nCommon variables:\n"
	@printf "  RAW_DIR=%s\n  DATASET=%s\n  CHECKPOINT=<local .pth>\n" "$(RAW_DIR)" "$(DATASET)"
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

test: ## Run the complete test suite.
	$(UV) run $(PYTHON) -m pytest $(TEST_DIR) -q $(PYTEST_ARGS)

test-verbose: ## Run tests with verbose output.
	$(UV) run $(PYTHON) -m pytest $(TEST_DIR) -vv $(PYTEST_ARGS)

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

hooks: ## Install both pre-commit and pre-push hooks.
	$(UV) run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
	@echo "Pre-commit and pre-push hooks installed."

hooks-run: ## Run commit-stage hooks on all tracked files.
	$(UV) run pre-commit run --hook-stage pre-commit --all-files

hooks-run-push: ## Run push-stage hooks on all tracked files.
	$(UV) run pre-commit run --hook-stage pre-push --all-files

hooks-uninstall: ## Remove the installed commit and push hooks.
	$(UV) run pre-commit uninstall --hook-type pre-commit
	$(UV) run pre-commit uninstall --hook-type pre-push

preprocess: ## Preprocess the full SHU-MI MATLAB archive into HDF5.
	$(UV) run $(PYTHON) -m main preprocess --raw-dir "$(RAW_DIR)" --output "$(DATASET)" $(PREPROCESS_FLAGS)

inspect-data: ## Validate shapes, classes, subject splits, leakage, and paper sample count.
	$(UV) run $(PYTHON) -m main inspect-data --data "$(DATASET)" $(INSPECT_STRICT_FLAG)

check-checkpoint: ## Download or validate the released CBraMod checkpoint and its SHA256.
	$(UV) run $(PYTHON) -m main check-checkpoint --config "$(CBRAMOD_CONFIG)" $(CHECKPOINT_FLAG)

train-cbramod: ## Run one CBraMod seed with the paper-aligned configuration.
	$(UV) run $(PYTHON) -m main train --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --output-dir "$(OUTPUT_DIR)/seed_3407" $(CHECKPOINT_FLAG) $(STRICT_DATA_FLAG)

train-simpleconv: ## Run one EEGSimpleConv seed on the same processed data.
	$(UV) run $(PYTHON) -m main train --config "$(SIMPLECONV_CONFIG)" --data "$(DATASET)" --output-dir "$(SIMPLECONV_OUTPUT_DIR)/seed_3407" $(STRICT_DATA_FLAG)

reproduce-cbramod: ## Run five seeds and write mean/std metrics to summary.json.
	$(UV) run $(PYTHON) -m main reproduce --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --output-dir "$(OUTPUT_DIR)" --seeds $(REPRO_SEEDS) $(CHECKPOINT_FLAG)

reproduce-simpleconv: ## Run five EEGSimpleConv seeds and aggregate the metrics.
	$(UV) run $(PYTHON) -m main reproduce --config "$(SIMPLECONV_CONFIG)" --data "$(DATASET)" --output-dir "$(SIMPLECONV_OUTPUT_DIR)" --seeds $(REPRO_SEEDS)

benchmark-cbramod: ## Benchmark CBraMod architecture without downloading weights.
	$(UV) run $(PYTHON) -m main benchmark --config "$(CBRAMOD_CONFIG)" --output "$(CBRAMOD_BENCHMARK)" --device "$(BENCHMARK_DEVICE)" --batch-sizes $(BENCHMARK_BATCHES) --warmup $(BENCHMARK_WARMUP) --iterations $(BENCHMARK_ITERATIONS) --random-init

benchmark-simpleconv: ## Benchmark EEGSimpleConv on the same device and input shape.
	$(UV) run $(PYTHON) -m main benchmark --config "$(SIMPLECONV_CONFIG)" --output "$(SIMPLECONV_BENCHMARK)" --device "$(BENCHMARK_DEVICE)" --batch-sizes $(BENCHMARK_BATCHES) --warmup $(BENCHMARK_WARMUP) --iterations $(BENCHMARK_ITERATIONS) --random-init

benchmark-models: benchmark-cbramod benchmark-simpleconv ## Benchmark both architectures.

compare-models: ## Generate Task C comparison.json and comparison.md.
	$(UV) run $(PYTHON) -m main compare --cbramod-summary "$(CBRAMOD_SUMMARY)" --simpleconv-summary "$(SIMPLECONV_SUMMARY)" --cbramod-benchmark "$(CBRAMOD_BENCHMARK)" --simpleconv-benchmark "$(SIMPLECONV_BENCHMARK)" --output-dir "$(TASK_C_OUTPUT_DIR)"

task-c: reproduce-simpleconv benchmark-models compare-models ## Run the complete EEGSimpleConv comparison workflow.

reproduce-cbramod-debug: ## Run the reproduction command without requiring all 25 subjects.
	$(UV) run $(PYTHON) -m main reproduce --config "$(CBRAMOD_CONFIG)" --data "$(DATASET)" --output-dir "$(OUTPUT_DIR)" --seeds $(REPRO_SEEDS) $(CHECKPOINT_FLAG) --allow-incomplete-data

sample-preprocess: ## Preprocess the bundled subject-1 sample for pipeline validation.
	$(UV) run $(PYTHON) -m main preprocess --raw-dir resources/shu-mi_dataset/mat_files --output data/processed/shu_mi_sample.h5 --overwrite

sample-inspect: ## Inspect the bundled incomplete sample (warnings are expected).
	$(UV) run $(PYTHON) -m main inspect-data --data data/processed/shu_mi_sample.h5

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

clean-data: ## Remove generated HDF5 data only; raw data is preserved.
	rm -rf data/processed

distclean: clean ## Remove caches plus the local virtual environment.
	rm -rf .venv
