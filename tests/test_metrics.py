import torch

from cbramod_experiments.utils import binary_metrics_from_logits


def test_perfect_binary_metrics() -> None:
    targets = torch.tensor([0, 1, 0, 1])
    logits = torch.tensor([-4.0, 4.0, -2.0, 2.0])
    metrics = binary_metrics_from_logits(logits, targets)
    assert metrics.balanced_accuracy == 1.0
    assert metrics.auroc == 1.0
    assert metrics.auprc == 1.0


def test_auprc_matches_trapezoidal_official_evaluator() -> None:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve

    targets = torch.tensor([0, 1, 0, 1, 0])
    probabilities = torch.tensor([0.90, 0.80, 0.70, 0.20, 0.10])
    logits = torch.logit(probabilities)
    metrics = binary_metrics_from_logits(logits, targets)
    precision, recall, _ = precision_recall_curve(targets.numpy(), probabilities.numpy())
    expected_auc = auc(recall, precision)
    expected_ap = average_precision_score(targets.numpy(), probabilities.numpy())
    assert metrics.auprc == expected_auc
    assert metrics.average_precision == expected_ap
    assert metrics.auprc != metrics.average_precision
