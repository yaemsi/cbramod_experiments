import torch

from cbramod_experiments.utils import binary_metrics_from_logits


def test_perfect_binary_metrics() -> None:
    targets = torch.tensor([0, 1, 0, 1])
    logits = torch.tensor([-4.0, 4.0, -2.0, 2.0])
    metrics = binary_metrics_from_logits(logits, targets)
    assert metrics.balanced_accuracy == 1.0
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0
