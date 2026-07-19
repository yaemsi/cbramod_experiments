from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from cbramod_experiments.models.cbramod import CBraModBackbone, CBraModClassifier


def _official_key(key: str) -> str:
    replacements = (
        ("patch_embedding.time_projection.", "patch_embedding.proj_in."),
        ("patch_embedding.spectral_projection.", "patch_embedding.spectral_proj."),
        (
            "patch_embedding.positional_encoding.",
            "patch_embedding.positional_encoding.0.",
        ),
        ("encoder.", "encoder.layers."),
        (".spatial_attention.", ".self_attn_s."),
        (".temporal_attention.", ".self_attn_t."),
        ("proj_out.", "proj_out.0."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def test_official_checkpoint_key_mapping(tmp_path: Path) -> None:
    source = CBraModBackbone(num_layers=1)
    official_state = {
        _official_key(name): value.detach().clone()
        for name, value in source.state_dict().items()
    }
    checkpoint = tmp_path / "pretrained_weights.pth"
    torch.save(official_state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    loaded = CBraModClassifier(
        num_layers=1,
        pretrained=True,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=digest,
        classifier="avg_pool",
    )
    loaded_state = loaded.backbone.state_dict()
    for name, expected in source.state_dict().items():
        if name.startswith("proj_out."):
            continue
        torch.testing.assert_close(loaded_state[name], expected)
    assert loaded.pretrained_checkpoint == str(checkpoint.resolve())
    assert loaded.pretrained_checkpoint_sha256 == digest
