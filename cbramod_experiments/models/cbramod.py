from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from einops.layers.torch import Rearrange
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.nn import functional as F


class CrissCrossEncoderLayer(nn.Module):
    """CBraMod layer with separate spatial and temporal attention."""

    def __init__(
        self,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        if d_model % 2 or nhead % 2:
            raise ValueError("d_model and nhead must be even for criss-cross attention")
        self.spatial_attention = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, source: Tensor) -> Tensor:
        x = source + self._attention_block(self.norm1(source))
        x = x + self._feedforward_block(self.norm2(x))
        return x

    def _attention_block(self, x: Tensor) -> Tensor:
        batch, channels, patches, width = x.shape
        spatial, temporal = x[..., : width // 2], x[..., width // 2 :]

        spatial = spatial.transpose(1, 2).reshape(batch * patches, channels, width // 2)
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[
            0
        ]
        spatial = spatial.reshape(batch, patches, channels, width // 2).transpose(1, 2)

        temporal = temporal.reshape(batch * channels, patches, width // 2)
        temporal = self.temporal_attention(
            temporal, temporal, temporal, need_weights=False
        )[0]
        temporal = temporal.reshape(batch, channels, patches, width // 2)
        return self.dropout1(torch.cat((spatial, temporal), dim=-1))

    def _feedforward_block(self, x: Tensor) -> Tensor:
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 200, d_model: int = 200) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.time_projection = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_projection = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, d_model),
            nn.Dropout(0.1),
        )
        self.positional_encoding = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=(19, 7),
            padding=(9, 3),
            groups=d_model,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, channels, patches, patch_size = x.shape
        if patch_size != self.patch_size:
            raise ValueError(
                f"Expected patch size {self.patch_size}, received {patch_size}"
            )
        if mask is None:
            masked = x
        else:
            masked = x.clone()
            masked[mask.bool()] = self.mask_encoding
        flattened = masked.reshape(batch, 1, channels * patches, patch_size)

        time_embedding = self.time_projection(flattened)
        time_embedding = time_embedding.permute(0, 2, 1, 3).reshape(
            batch, channels, patches, self.d_model
        )

        spectrum = torch.fft.rfft(
            flattened.reshape(batch * channels * patches, patch_size),
            dim=-1,
            norm="forward",
        ).abs()
        spectrum = spectrum.reshape(batch, channels, patches, patch_size // 2 + 1)
        embedding = time_embedding + self.spectral_projection(spectrum)
        position = self.positional_encoding(embedding.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        return embedding + position


class CBraModBackbone(nn.Module):
    """Architecture compatible with the authors' released foundation checkpoint."""

    def __init__(
        self,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        num_layers: int = 12,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, d_model=d_model)
        layer = CrissCrossEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.proj_out: nn.Module = nn.Linear(d_model, d_model)
        self.apply(_weights_init)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.patch_embedding(x, mask)
        for layer in self.encoder:
            x = layer(x)
        return self.proj_out(x)


class CBraModClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int = 32,
        num_patches: int = 4,
        d_model: int = 200,
        dim_feedforward: int = 800,
        num_layers: int = 12,
        nhead: int = 8,
        classifier: str = "all_patch_reps",
        dropout: float = 0.1,
        pretrained: bool = False,
        checkpoint_repo: str = "weighting666/CBraMod",
        checkpoint_filename: str = "pretrained_weights.pth",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CBraModBackbone(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            nhead=nhead,
        )
        if pretrained:
            self.load_pretrained(checkpoint_repo, checkpoint_filename)
        self.backbone.proj_out = nn.Identity()
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        if classifier == "avg_pool":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(d_model, 1),
            )
        elif classifier == "linear":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(num_channels * num_patches * d_model, 1),
            )
        elif classifier == "all_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(num_channels * num_patches * d_model, 4 * d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            raise ValueError(f"Unknown CBraMod classifier: {classifier}")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, channels, time], received {tuple(x.shape)}"
            )
        batch, channels, time = x.shape
        if time % 200:
            raise ValueError(
                "CBraMod expects a time length divisible by its 200-sample patch size"
            )
        patches = time // 200
        features = self.backbone(x.reshape(batch, channels, patches, 200))
        return self.classifier(features).reshape(-1)

    def load_pretrained(self, repo_id: str, filename: str) -> None:
        checkpoint = hf_hub_download(repo_id=repo_id, filename=filename)
        payload: Any = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=True
        )
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(payload)!r}")
        payload = {_remap_official_key(key): value for key, value in payload.items()}
        missing, unexpected = self.backbone.load_state_dict(payload, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Released checkpoint is not architecture-compatible. "
                f"Missing keys: {missing}; unexpected keys: {unexpected}"
            )


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _remap_official_key(key: str) -> str:
    """Map the authors' module names to the cleaned implementation names."""
    replacements = (
        ("patch_embedding.proj_in.", "patch_embedding.time_projection."),
        ("patch_embedding.spectral_proj.", "patch_embedding.spectral_projection."),
        (
            "patch_embedding.positional_encoding.0.",
            "patch_embedding.positional_encoding.",
        ),
        ("encoder.layers.", "encoder."),
        (".self_attn_s.", ".spatial_attention."),
        (".self_attn_t.", ".temporal_attention."),
        ("proj_out.0.", "proj_out."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key
