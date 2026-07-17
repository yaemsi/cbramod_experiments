# ---------------------------------------------------------------------------
# market_data_runner — dev commands
# Run with: make <target>   |   make help for a list of targets
# ---------------------------------------------------------------------------

.PHONY: main test install lint format typecheck hooks clean


all: test

## Show available targets
help:
	@echo ""
	@echo "Usage: make <target>"



## Run the smoke test — quick install check (~15s)
install:
	uv sync
	uv pip install .

## Run the smoke test — quick install check (~15s)
main:
	uv run python -m main smoke

## Fast correctness suite only 
test:
	uv run pytest tests/*.py -v

## Run ruff linter (auto-fixes safe issues)
lint:
	uv run ruff check cbramod_experiments tests --fix

## Run ruff formatter
format:
	uv run ruff format cbramod_experiments tests

## Run pyright type checker
typecheck:
	uv run pyright

## Install pre-commit hooks (run once after cloning)
hooks:
	uv run pre-commit install
	@echo "Pre-commit hooks installed. They will run automatically before each commit."

## Remove all generated artefacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"        -delete 2>/dev/null || true
	find . -type f -name "*.pyo"        -delete 2>/dev/null || true
	find . -type f -name "*.png"        -delete 2>/dev/null || true
	rm -rf ./outputs/ 2>/dev/null || true