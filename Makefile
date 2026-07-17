# CBraMod homework development commands.
# Examples:
#   make sync
#   make test
#   make preprocess RAW_DIR=/path/to/shu DATASET=data/processed/shu_mi.h5
#   make train-cbramod DATASET=data/processed/shu_mi.h5

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python
PACKAGE := cbramod_experiments
TEST_DIR := tests

RAW_DIR ?= data/raw/shu
DATASET ?= data/processed/shu_mi.h5
CBRAMOD_CONFIG ?= configs/cbramod.yaml
SIMPLECONV_CONFIG ?= configs/eegsimpleconv.yaml
PYTEST_ARGS ?=
OVERWRITE ?= 0

ifeq ($(OVERWRITE),1)
PREPROCESS_FLAGS := --overwrite
else
PREPROCESS_FLAGS :=
endif

.PHONY: help all sync install lock update smoke main test test-verbose \
        lint lint-fix format format-check typecheck check ci hooks \
        preprocess train-cbramod train-simpleconv clean distclean

all: check ## Run the full local validation suite.

help: ## Show available targets.
	@printf "\nUsage: make <target> [VARIABLE=value]\n\n"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nCommon variables:\n"
	@printf "  RAW_DIR=%s\n  DATASET=%s\n  PYTEST_ARGS=%s\n  OVERWRITE=0|1\n\n" "$(RAW_DIR)" "$(DATASET)" "$(PYTEST_ARGS)"

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

hooks: ## Install pre-commit hooks.
	$(UV) run pre-commit install
	@echo "Pre-commit hooks installed."

preprocess: ## Preprocess SHU-MI MATLAB files into HDF5.
	$(UV) run $(PYTHON) -m main preprocess --raw-dir "$(RAW_DIR)" --output "$(DATASET)" $(PREPROCESS_FLAGS)

train-cbramod: ## Train/fine-tune CBraMod.
	$(UV) run $(PYTHON) -m main train --config "$(CBRAMOD_CONFIG)"

train-simpleconv: ## Train EEGSimpleConv.
	$(UV) run $(PYTHON) -m main train --config "$(SIMPLECONV_CONFIG)"

clean: ## Remove caches and generated experiment outputs.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf build dist outputs

distclean: clean ## Remove caches plus the local virtual environment.
	rm -rf .venv
