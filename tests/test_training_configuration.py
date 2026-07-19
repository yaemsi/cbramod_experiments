import torch
from torch import nn

from cbramod_experiments.utils.train import _build_optimizer, _build_scheduler


def test_simpleconv_optimizer_and_epoch_step_schedule() -> None:
    model = nn.Linear(4, 1)
    optimizer = _build_optimizer(
        model,
        lr=1e-3,
        head_lr=None,
        weight_decay=0.0,
        optimizer_name="adam",
    )
    assert isinstance(optimizer, torch.optim.Adam)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name="step",
        epochs=50,
        steps_per_epoch=10,
        min_lr=0.0,
        lr_decay_epoch=40,
        lr_decay_gamma=0.1,
        scheduler_interval="epoch",
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    for _ in range(40):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1e-4
