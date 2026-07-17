from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import evaluate_binary
from .metrics import BinaryMetrics
from .utils import save_json


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    validation: BinaryMetrics
    test: BinaryMetrics
    checkpoint_path: str


def fit_binary_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    head_lr: float | None,
    weight_decay: float,
    grad_clip_norm: float,
    patience: int,
    amp: bool,
    output_dir: str | Path,
) -> FitResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = _build_optimizer(model, lr=lr, head_lr=head_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(train_loader)), eta_min=1e-6
    )
    use_amp = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: BinaryMetrics | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for signals, targets in progress:
            signals = signals.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(signals).reshape(-1)
                loss = F.binary_cross_entropy_with_logits(logits, targets.reshape(-1))
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = targets.numel()
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        val_metrics = evaluate_binary(model, val_loader, device)
        train_loss = total_loss / max(1, total_examples)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_balanced_accuracy": val_metrics.balanced_accuracy,
                "val_auprc": val_metrics.auprc,
                "val_auroc": val_metrics.auroc,
            }
        )
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"val_bacc={val_metrics.balanced_accuracy:.4f} "
            f"val_auprc={val_metrics.auprc:.4f} val_auroc={val_metrics.auroc:.4f}"
        )

        if best_metrics is None or val_metrics.auroc > best_metrics.auroc:
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training completed without producing a checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate_binary(model, test_loader, device)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
        },
        checkpoint_path,
    )
    save_json({"history": history}, output_dir / "history.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "validation": best_metrics.to_dict(),
            "test": test_metrics.to_dict(),
        },
        output_dir / "metrics.json",
    )
    return FitResult(best_epoch, best_metrics, test_metrics, str(checkpoint_path))


def _build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    head_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if head_lr is None or not hasattr(model, "backbone"):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
