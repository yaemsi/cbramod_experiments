from __future__ import annotations

from collections.abc import Iterable

import torch

from .metrics import BinaryMetrics, binary_metrics_from_logits


@torch.inference_mode()
def evaluate_binary(
    model: torch.nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> BinaryMetrics:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for signals, targets in data_loader:
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    if not all_logits:
        raise ValueError("Cannot evaluate an empty data loader")
    return binary_metrics_from_logits(torch.cat(all_logits), torch.cat(all_targets))
