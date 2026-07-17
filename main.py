from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cbramod_experiments.utils import (
    load_config,
    binary_metrics_from_logits,
    count_parameters, 
    resolve_device, 
    seed_everything,
    fit_binary_classifier,
)

from cbramod_experiments.datasets import (
    SHUDataModule, 
    preprocess_shu
)

from cbramod_experiments.models import (
    build_model, 
    CBraModClassifier,
    EEGSimpleConv,
)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod SHU-MI homework experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Convert SHU .mat files to HDF5")
    preprocess_parser.add_argument("--raw-dir", required=True)
    preprocess_parser.add_argument("--output", required=True)
    preprocess_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train a configured model")
    train_parser.add_argument("--config", required=True)

    subparsers.add_parser("smoke", help="Run CPU-friendly model and metric smoke tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preprocess":
        summary = preprocess_shu(args.raw_dir, args.output, overwrite=args.overwrite)
        print(summary)
    elif args.command == "train":
        run_training(args.config)
    elif args.command == "smoke":
        run_smoke_test()


def run_training(config_path: str | Path) -> None:
    config = load_config(config_path)
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    model = build_model(config.model)
    data = SHUDataModule(
        config.data.path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        seed=config.training.seed,
    ).loaders()
    print(
        f"model={config.model.name} device={device} "
        f"parameters={count_parameters(model):,} trainable={count_parameters(model, True):,}"
    )
    result = fit_binary_classifier(
        model,
        data["train"],
        data["val"],
        data["test"],
        device,
        epochs=config.training.epochs,
        lr=config.training.lr,
        head_lr=config.training.head_lr,
        weight_decay=config.training.weight_decay,
        grad_clip_norm=config.training.grad_clip_norm,
        patience=config.training.patience,
        amp=config.training.amp,
        output_dir=config.training.output_dir,
    )
    print(result)


def run_smoke_test() -> None:
    seed_everything(7)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    cbramod = CBraModClassifier(
        num_channels=2, num_patches=1, num_layers=1, pretrained=False, classifier="avg_pool"
    )
    cbramod_input = torch.randn(1, 2, 200)
    cbramod_logits = cbramod(cbramod_input)
    cbramod_logits.sum().backward()
    print(f"cbramod: logits={tuple(cbramod_logits.shape)}")

    simpleconv = EEGSimpleConv(num_channels=4, num_blocks=1)
    simpleconv_input = torch.randn(2, 4, 800)
    simpleconv_logits = simpleconv(simpleconv_input)
    simpleconv_logits.sum().backward()
    print(f"eegsimpleconv: logits={tuple(simpleconv_logits.shape)}")

    metrics = binary_metrics_from_logits(
        torch.tensor([-2.0, 2.0, -1.0, 1.0, -0.5, 0.5, -3.0, 3.0]), targets
    )
    print(f"metrics: {metrics.to_dict()}")


if __name__ == "__main__":
    main()
