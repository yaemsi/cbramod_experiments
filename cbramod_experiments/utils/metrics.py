from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    balanced_accuracy: float
    auprc: float
    auroc: float
    confusion_matrix: list[list[int]]
    num_examples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def binary_metrics_from_logits(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute SHU-MI metrics from either one-logit or two-logit predictions."""
    logits_np = _to_numpy(logits)
    targets_np = _to_numpy(targets).reshape(-1).astype(np.int64)

    if logits_np.ndim == 2 and logits_np.shape[1] == 2:
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        scores = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    else:
        logits_np = logits_np.reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -80.0, 80.0)))

    if scores.shape[0] != targets_np.shape[0]:
        raise ValueError(f"Mismatched predictions and targets: {scores.shape} vs {targets_np.shape}")
    if np.unique(targets_np).size < 2:
        raise ValueError("AUROC/AUPRC require both classes in the evaluated split")

    predictions = (scores >= threshold).astype(np.int64)
    return BinaryMetrics(
        balanced_accuracy=float(balanced_accuracy_score(targets_np, predictions)),
        auprc=float(average_precision_score(targets_np, scores)),
        auroc=float(roc_auc_score(targets_np, scores)),
        confusion_matrix=confusion_matrix(targets_np, predictions, labels=[0, 1]).tolist(),
        num_examples=int(targets_np.size),
    )


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
