import torch
from torch import nn

from cbramod_experiments.models import CBraModClassifier, EEGSimpleConv


def test_cbramod_output_shape() -> None:
    model = CBraModClassifier(num_layers=1, pretrained=False, classifier="avg_pool")
    output = model(torch.randn(2, 32, 800))
    assert output.shape == (2,)


def test_eegsimpleconv_output_shape() -> None:
    model = EEGSimpleConv(num_blocks=1)
    output = model(torch.randn(2, 32, 800))
    assert output.shape == (2,)


def test_eegsimpleconv_matches_upstream_channel_growth() -> None:
    model = EEGSimpleConv(feature_maps=128, num_blocks=2)
    second_block_first_conv = model.blocks[1][0]
    assert isinstance(second_block_first_conv, nn.Conv1d)
    assert second_block_first_conv.in_channels == 128
    assert second_block_first_conv.out_channels == 180
    assert model.feature_dim == 180


def test_eegsimpleconv_can_return_features() -> None:
    model = EEGSimpleConv(num_blocks=2, return_features=True)
    features = model(torch.randn(2, 32, 800))
    assert features.shape == (2, 180)
