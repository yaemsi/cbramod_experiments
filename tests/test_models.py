import torch

from cbramod_experiments.models import (
    CBraModClassifier, 
    EEGSimpleConv,
)


def test_cbramod_output_shape() -> None:
    model = CBraModClassifier(num_layers=1, pretrained=False, classifier="avg_pool")
    output = model(torch.randn(2, 32, 800))
    assert output.shape == (2,)


def test_eegsimpleconv_output_shape() -> None:
    model = EEGSimpleConv(num_blocks=1)
    output = model(torch.randn(2, 32, 800))
    assert output.shape == (2,)
