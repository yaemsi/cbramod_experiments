from __future__ import annotations

import argparse
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from cbramod_experiments.datasets import SHUDataModule, audit_shu_h5, preprocess_shu
from cbramod_experiments.models import CBraModClassifier, EEGSimpleConv, build_model
from cbramod_experiments.utils import (
    CBRAMOD_PAPER_REFERENCE,
    ExperimentConfig,
    FitResult,
    ReproductionRun,
    aggregate_reproduction_runs,
    benchmark_model,
    binary_metrics_from_logits,
    compare_experiments,
    count_parameters,
    fit_binary_classifier,
    load_config,
    resolve_device,
    save_json,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod SHU-MI homework experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Convert SHU .mat files to HDF5"
    )
    preprocess_parser.add_argument("--raw-dir", required=True)
    preprocess_parser.add_argument("--output", required=True)
    preprocess_parser.add_argument("--overwrite", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="Audit a processed SHU-MI HDF5 file"
    )
    inspect_parser.add_argument("--data", required=True)
    inspect_parser.add_argument("--strict", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train one configured model")
    _add_run_arguments(train_parser)
    train_parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Require the complete 25-subject SHU-MI protocol before training",
    )

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Run and aggregate a multi-seed SHU-MI experiment"
    )
    _add_run_arguments(reproduce_parser)
    reproduce_parser.add_argument(
        "--seeds", type=int, nargs="+", default=[3407, 3408, 3409, 3410, 3411]
    )
    reproduce_parser.add_argument(
        "--allow-incomplete-data",
        action="store_true",
        help="Only for pipeline debugging; never use for reported results",
    )

    checkpoint_parser = subparsers.add_parser(
        "check-checkpoint", help="Download/load the released CBraMod checkpoint"
    )
    checkpoint_parser.add_argument("--config", default="configs/cbramod.yaml")
    checkpoint_parser.add_argument("--checkpoint-path")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Measure model size, latency, throughput, and peak memory"
    )
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--output", required=True)
    benchmark_parser.add_argument("--device", default="auto")
    benchmark_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    benchmark_parser.add_argument("--iterations", type=int, default=100)
    benchmark_parser.add_argument(
        "--random-init",
        action="store_true",
        help="Avoid loading pretrained weights; appropriate for architecture timing",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Create Task C JSON and Markdown comparison reports"
    )
    compare_parser.add_argument("--cbramod-summary", required=True)
    compare_parser.add_argument("--simpleconv-summary", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--cbramod-benchmark")
    compare_parser.add_argument("--simpleconv-benchmark")

    subparsers.add_parser("smoke", help="Run CPU-friendly model and metric smoke tests")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-path")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preprocess":
        summary = preprocess_shu(args.raw_dir, args.output, overwrite=args.overwrite)
        print(summary)
    elif args.command == "inspect-data":
        audit = audit_shu_h5(args.data, require_complete_protocol=args.strict)
        print_audit(audit.to_dict())
    elif args.command == "train":
        config = _load_with_overrides(args)
        result = execute_training(config, strict_data=args.strict_data)
        print(result.to_dict())
    elif args.command == "reproduce":
        run_reproduction(args)
    elif args.command == "check-checkpoint":
        check_checkpoint(args.config, args.checkpoint_path)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "compare":
        result = compare_experiments(
            args.cbramod_summary,
            args.simpleconv_summary,
            args.output_dir,
            cbramod_benchmark_path=args.cbramod_benchmark,
            simpleconv_benchmark_path=args.simpleconv_benchmark,
        )
        print(result["metrics"])
    elif args.command == "smoke":
        run_smoke_test()


def _load_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config)
    data = replace(config.data, path=args.data) if args.data else config.data
    model = (
        replace(config.model, checkpoint_path=args.checkpoint_path)
        if args.checkpoint_path
        else config.model
    )
    training = config.training
    if args.seed is not None:
        training = replace(training, seed=args.seed)
    if args.output_dir:
        training = replace(training, output_dir=args.output_dir)
    return replace(config, data=data, model=model, training=training)


def execute_training(config: ExperimentConfig, *, strict_data: bool) -> FitResult:
    audit = audit_shu_h5(config.data.path, require_complete_protocol=strict_data)
    print_audit(audit.to_dict())
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
        f"model={config.model.name} seed={config.training.seed} device={device} "
        f"parameters={count_parameters(model):,} "
        f"trainable={count_parameters(model, True):,}"
    )
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.to_dict(), output_dir / "resolved_config.json")
    save_json(
        _runtime_manifest(model, device, audit.to_dict()), output_dir / "run.json"
    )
    return fit_binary_classifier(
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
        output_dir=output_dir,
        optimizer_name=config.training.optimizer,
        scheduler_name=config.training.scheduler,
        scheduler_interval=config.training.scheduler_interval,
        min_lr=config.training.min_lr,
        lr_decay_epoch=config.training.lr_decay_epoch,
        lr_decay_gamma=config.training.lr_decay_gamma,
        label_smoothing=config.training.label_smoothing,
    )


def run_reproduction(args: argparse.Namespace) -> None:
    base = _load_with_overrides(args)
    strict = not args.allow_incomplete_data
    audit_shu_h5(base.data.path, require_complete_protocol=strict)
    output_root = Path(args.output_dir or base.training.output_dir)
    runs: list[ReproductionRun] = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        config = replace(
            base,
            training=replace(base.training, seed=seed, output_dir=str(run_dir)),
        )
        result = execute_training(config, strict_data=strict)
        runs.append(ReproductionRun(seed=seed, output_dir=str(run_dir), result=result))

    is_cbramod = base.model.name.lower() == "cbramod"
    summary = aggregate_reproduction_runs(
        runs,
        output_root / "summary.json",
        model_name=base.model.name,
        paper_reference=CBRAMOD_PAPER_REFERENCE if is_cbramod else None,
    )
    print(summary["test_aggregate"])


def check_checkpoint(config_path: str, checkpoint_path: str | None) -> None:
    config = load_config(config_path)
    model_config = config.model
    if checkpoint_path is not None:
        model_config = replace(model_config, checkpoint_path=checkpoint_path)
    if model_config.name.lower() != "cbramod":
        raise ValueError("check-checkpoint requires a CBraMod configuration")
    model_config = replace(model_config, pretrained=True)
    model = build_model(model_config)
    if not isinstance(model, CBraModClassifier):
        raise TypeError("Expected a CBraModClassifier")
    print(
        {
            "checkpoint": model.pretrained_checkpoint,
            "sha256": model.pretrained_checkpoint_sha256,
            "parameters": count_parameters(model),
        }
    )


def run_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model_config = config.model
    if args.random_init:
        model_config = replace(model_config, pretrained=False, checkpoint_path=None)
    model = build_model(model_config)
    device = resolve_device(args.device)
    result = benchmark_model(
        model,
        model_name=model_config.name,
        device=device,
        num_channels=model_config.num_channels,
        num_points=int(model_config.sampling_rate * model_config.window_seconds),
        batch_sizes=args.batch_sizes,
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
        output_path=args.output,
    )
    print(result.to_dict())


def print_audit(audit: dict[str, Any]) -> None:
    print(
        "dataset="
        f"{audit['path']} examples={audit['examples']} channels={audit['channels']} "
        f"points={audit['points']} subjects_complete={audit['complete_subject_protocol']} "
        f"paper_ready={audit['paper_protocol_ready']}"
    )
    print(f"split_examples={audit['split_examples']}")
    print(f"split_class_counts={audit['split_class_counts']}")
    for warning in audit["warnings"]:
        print(f"WARNING: {warning}")


def _runtime_manifest(
    model: torch.nn.Module, device: torch.device, audit: dict[str, Any]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, True),
        "dataset_audit": audit,
    }
    if isinstance(model, CBraModClassifier):
        manifest["pretrained_checkpoint"] = model.pretrained_checkpoint
        manifest["pretrained_checkpoint_sha256"] = model.pretrained_checkpoint_sha256
    return manifest


def run_smoke_test() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed_everything(7)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    cbramod = CBraModClassifier(
        num_channels=2,
        num_patches=1,
        num_layers=1,
        pretrained=False,
        classifier="avg_pool",
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
