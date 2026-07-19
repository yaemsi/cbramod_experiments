"""Configuration, training, evaluation, benchmarking, and comparison utilities."""

from .benchmark import BatchBenchmark, ModelBenchmark, benchmark_model
from .compare import compare_experiments
from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from .evaluate import evaluate_binary
from .metrics import BinaryMetrics, binary_metrics_from_logits
from .reproduce import (
    CBRAMOD_PAPER_REFERENCE,
    PAPER_REFERENCE,
    ReproductionRun,
    aggregate_reproduction_runs,
)
from .train import FitResult, fit_binary_classifier
from .utils import (
    count_parameters,
    resolve_device,
    save_json,
    seed_everything,
    seed_worker,
)

__all__ = [
    "BatchBenchmark",
    "BinaryMetrics",
    "CBRAMOD_PAPER_REFERENCE",
    "DataConfig",
    "ExperimentConfig",
    "FitResult",
    "ModelBenchmark",
    "ModelConfig",
    "PAPER_REFERENCE",
    "ReproductionRun",
    "TrainingConfig",
    "aggregate_reproduction_runs",
    "benchmark_model",
    "binary_metrics_from_logits",
    "compare_experiments",
    "count_parameters",
    "evaluate_binary",
    "fit_binary_classifier",
    "load_config",
    "resolve_device",
    "save_json",
    "seed_everything",
    "seed_worker",
]
