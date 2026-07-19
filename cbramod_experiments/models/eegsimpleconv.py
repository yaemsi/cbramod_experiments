from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn
from torchaudio.transforms import Resample


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class EEGSimpleConv(nn.Module):
    """EEG-SimpleConv adapted to the shared binary-classification interface.

    The convolutional backbone follows the public Braindecode implementation:
    input resampling, an initial temporal convolution, repeated two-convolution
    blocks with max pooling, global temporal average pooling, and a linear head.
    A single output logit is used so the model shares CBraMod's loss and metrics.
    """

    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 200,
        feature_maps: int = 128,
        num_blocks: int = 2,
        resampling_rate: int = 80,
        kernel_size: int = 8,
        activation: str = "relu",
        return_features: bool = False,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if sampling_rate <= 0 or resampling_rate <= 0:
            raise ValueError("sampling rates must be positive")
        if feature_maps <= 0 or num_blocks <= 0 or kernel_size <= 0:
            raise ValueError("feature_maps, num_blocks, and kernel_size must be positive")
        try:
            activation_factory = _ACTIVATIONS[activation.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported activation {activation!r}; choose from {sorted(_ACTIVATIONS)}"
            ) from exc

        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate
        self.return_features = return_features
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
            activation_factory(),
        )

        blocks: list[nn.Module] = []
        in_features = feature_maps
        out_features = feature_maps
        for index in range(num_blocks):
            if index > 0:
                # Preserve the upstream implementation exactly. Using sqrt(2)
                # changes 128 -> 181, whereas int(1.414 * 128) gives 180.
                out_features = int(1.414 * out_features)
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
                    activation_factory(),
                    nn.Conv1d(
                        out_features,
                        out_features,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_features),
                    activation_factory(),
                )
            )
            in_features = out_features

        self.blocks = nn.ModuleList(blocks)
        self.feature_dim = out_features
        self.final_layer = nn.Linear(out_features, 1)

    def extract_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, channels, time], received {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, received {x.shape[1]}"
            )
        features = self.stem(self.resample(x.contiguous()))
        for block in self.blocks:
            features = block(features)
        return features.mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.extract_features(x)
        if self.return_features:
            return features
        return self.final_layer(features).reshape(-1)
