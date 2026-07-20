from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

    def to_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "checkpoint_path": self.checkpoint_path,
        }


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
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
    scheduler_interval: Literal["step", "epoch"] = "step",
    min_lr: float = 1e-6,
    lr_decay_epoch: int = 40,
    lr_decay_gamma: float = 0.1,
    label_smoothing: float = 0.0,
) -> FitResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = _build_optimizer(
        model,
        lr=lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        min_lr=min_lr,
        lr_decay_epoch=lr_decay_epoch,
        lr_decay_gamma=lr_decay_gamma,
        scheduler_interval=scheduler_interval,
    )
    use_amp = amp and device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_amp)

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
            targets = targets.to(device, non_blocking=True).float().reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(signals).reshape(-1)
                smoothed_targets = _smooth_binary_targets(targets, label_smoothing)
                loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets)
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_interval == "step":
                scheduler.step()
            batch_size = targets.numel()
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        val_metrics = evaluate_binary(model, val_loader, device)
        train_loss = total_loss / max(1, total_examples)
        backbone_lr = float(optimizer.param_groups[0]["lr"])
        current_head_lr = float(optimizer.param_groups[-1]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "backbone_lr": backbone_lr,
                "head_lr": current_head_lr,
                "val_balanced_accuracy": val_metrics.balanced_accuracy,
                "val_auprc": val_metrics.auprc,
                "val_auroc": val_metrics.auroc,
                "val_average_precision": val_metrics.average_precision,
            }
        )
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"val_bacc={val_metrics.balanced_accuracy:.4f} "
            f"val_auprc={val_metrics.auprc:.4f} "
            f"val_auroc={val_metrics.auroc:.4f}"
        )

        if best_metrics is None or val_metrics.auroc > best_metrics.auroc:
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training completed without producing a checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
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
    result = FitResult(best_epoch, best_metrics, test_metrics, str(checkpoint_path))
    save_json(result.to_dict(), output_dir / "metrics.json")
    return result


def _smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    head_lr: float | None,
    weight_decay: float,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    optimizer_key = optimizer_name.lower()
    optimizer_type: type[torch.optim.Optimizer]
    if optimizer_key == "adam":
        optimizer_type = torch.optim.Adam
    elif optimizer_key == "adamw":
        optimizer_type = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    backbone_module = getattr(model, "backbone", None)
    if head_lr is None or not isinstance(backbone_module, torch.nn.Module):
        return optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)

    backbone = [p for p in backbone_module.parameters() if p.requires_grad]
    head = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("backbone.") and p.requires_grad
    ]
    groups: list[dict[str, object]] = []
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return optimizer_type(groups, weight_decay=weight_decay)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    min_lr: float,
    lr_decay_epoch: int,
    lr_decay_gamma: float,
    scheduler_interval: Literal["step", "epoch"],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_key = scheduler_name.lower()
    if scheduler_key == "none":
        return None
    if scheduler_key == "cosine":
        total_units = (
            epochs * steps_per_epoch if scheduler_interval == "step" else epochs
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_units), eta_min=min_lr
        )
    if scheduler_key == "step":
        step_size = (
            lr_decay_epoch * steps_per_epoch
            if scheduler_interval == "step"
            else lr_decay_epoch
        )
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, step_size), gamma=lr_decay_gamma
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
