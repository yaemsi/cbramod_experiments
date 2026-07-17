from __future__ import annotations

import math

from torch import Tensor, nn
from torchaudio.transforms import Resample


class EEGSimpleConv(nn.Module):
    """Simple 1D convolutional motor-imagery baseline."""

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 200,
        feature_maps: int = 128,
        num_blocks: int = 2,
        resampling_rate: int = 80,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()
        self.resample = (
            Resample(orig_freq=sampling_rate, new_freq=resampling_rate)
            if sampling_rate != resampling_rate
            else nn.Identity()
        )
        self.stem = nn.Sequential(
            nn.Conv1d(
                num_channels,
                feature_maps,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(feature_maps),
            nn.ReLU(),
        )
        blocks: list[nn.Module] = []
        in_features = feature_maps
        out_features = feature_maps
        for index in range(num_blocks):
            if index > 0:
                out_features = int(math.sqrt(2) * out_features)
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_features,
                        out_features,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_features),
                    nn.MaxPool1d(2),
                    nn.ReLU(),
                    nn.Conv1d(
                        out_features,
                        out_features,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                )
            )
            in_features = out_features
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(out_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, channels, time], received {tuple(x.shape)}"
            )
        features = self.stem(self.resample(x.contiguous()))
        for block in self.blocks:
            features = block(features)
        return self.classifier(features.mean(dim=-1)).reshape(-1)
