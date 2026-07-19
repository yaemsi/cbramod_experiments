from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cbramod_experiments.utils import fit_binary_classifier


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).reshape(-1)


def test_training_loop_writes_reproduction_artifacts(tmp_path: Path) -> None:
    signals = torch.tensor(
        [[-2.0, -1.0], [2.0, 1.0], [-1.0, -2.0], [1.0, 2.0]]
    )
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2, shuffle=False)
    result = fit_binary_classifier(
        TinyClassifier(),
        loader,
        loader,
        loader,
        torch.device("cpu"),
        epochs=2,
        lr=1e-2,
        head_lr=None,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        patience=2,
        amp=False,
        output_dir=tmp_path,
    )
    assert result.best_epoch in {1, 2}
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
