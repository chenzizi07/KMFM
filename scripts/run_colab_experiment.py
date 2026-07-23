from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DEFAULT = Path("/content/drive/MyDrive/Colab/Unsupervised/KMFM")
DATA_DEFAULT = Path("/content/drive/MyDrive/Colab/Datasets")

DATASETS = {
    "pavia_university": {
        "data": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
        "data_key": "paviaU",
        "gt_key": "paviaU_gt",
    },
    "houston2013": {
        "data": "Houston_data.mat",
        "gt": "Houston_gt.mat",
        "data_key": "hsi",
        "gt_key": "groundT",
    },
    "ksc": {
        "data": "KSC.mat",
        "gt": "KSC_gt.mat",
        "data_key": "KSC",
        "gt_key": "KSC_gt",
    },
    "botswana": {
        "data": "Botswana.mat",
        "gt": "Botswana_gt.mat",
        "data_key": "Botswana",
        "gt_key": "Botswana_gt",
    },
}

CORE_VARIANTS = [
    {"name": "lassf_mlp_concat_h64", "spectral": "mlp", "fusion": "concat"},
    {"name": "lassf_conv1d_concat_h64", "spectral": "conv1d", "fusion": "concat"},
    {"name": "lassf_conv1d_gate_h64", "spectral": "conv1d", "fusion": "gate"},
    {
        "name": "lassf_conv1d_reliability_h64",
        "spectral": "conv1d",
        "fusion": "reliability",
    },
]

ABLATION_VARIANTS = [
    {
        "name": "lassf_conv1d_spatial_only_h64",
        "spectral": "conv1d",
        "fusion": "spatial_only",
    },
    {
        "name": "lassf_conv1d_spectral_only_h64",
        "spectral": "conv1d",
        "fusion": "spectral_only",
    },
]

CALIBRATED_V3_VARIANTS = [
    {
        "name": "lassf_mlp_spatial_only_v3_h64",
        "spectral": "mlp",
        "fusion": "spatial_only",
        "normalize_branches": True,
    },
    {
        "name": "lassf_mlp_spectral_only_v3_h64",
        "spectral": "mlp",
        "fusion": "spectral_only",
        "normalize_branches": True,
    },
    {
        "name": "lassf_mlp_concat_norm_v3_h64",
        "spectral": "mlp",
        "fusion": "concat",
        "normalize_branches": True,
    },
    {
        "name": "lassf_mlp_global_norm_v3_h64",
        "spectral": "mlp",
        "fusion": "global",
        "normalize_branches": True,
    },
    {
        "name": "lassf_mlp_gate_norm_v3_h64",
        "spectral": "mlp",
        "fusion": "gate",
        "normalize_branches": True,
    },
    {
        "name": "lassf_mlp_entropy_softmax_v3_h64",
        "spectral": "mlp",
        "fusion": "entropy_softmax",
        "normalize_branches": True,
        "entropy_temperature": 0.25,
        "calibrate_branch_temperatures": True,
    },
]

OASD_V6_VARIANTS = [
    {
        "name": "lassf_mlp_spatial_only_v6_h64",
        "spectral": "mlp",
        "fusion": "spatial_only",
        "normalize_branches": True,
        "distillation": {"mode": "none", "coefficient": 0.0, "temperature": 2.0},
    },
    {
        "name": "lassf_mlp_uniform_distill_v6_h64",
        "spectral": "mlp",
        "fusion": "spatial_only",
        "normalize_branches": True,
        "distillation": {"mode": "uniform", "coefficient": 0.5, "temperature": 2.0},
    },
    {
        "name": "lassf_mlp_oof_adv_distill_v6_h64",
        "spectral": "mlp",
        "fusion": "spatial_only",
        "normalize_branches": True,
        "distillation": {
            "mode": "oof_class",
            "coefficient": 0.5,
            "temperature": 2.0,
            "folds": 3,
            "oof_epochs": 60,
            "oof_aux_weight": 0.5,
            "prior_strength": 4.0,
            "reference_gain": 0.25,
        },
    },
]


def _csv_values(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _seeds(raw: str) -> list[int]:
    try:
        values = [int(item) for item in _csv_values(raw)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return values


def _read_status(run_dir: Path) -> str | None:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        return str(json.loads(status_path.read_text(encoding="utf-8")).get("state"))
    except (OSError, json.JSONDecodeError):
        return "invalid"


def _archive_incomplete_run(
    run_dir: Path,
    *,
    results_root: Path,
    experiment: str,
    status: str | None,
) -> Path:
    experiment_root = results_root / experiment
    try:
        relative_run = run_dir.relative_to(experiment_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to archive a run outside the experiment root: {run_dir}"
        ) from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    state = status or "missing_status"
    archive_dir = (
        results_root
        / "_incomplete"
        / experiment
        / relative_run.parent
        / f"{relative_run.name}__{timestamp}__{state}"
    )
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.rename(archive_dir)
    return archive_dir


def _make_split(
    *,
    project_root: Path,
    data_path: Path,
    gt_path: Path,
    spec: dict[str, str],
    dataset: str,
    protocol: str,
    seed: int,
    args: argparse.Namespace,
) -> Path:
    split_path = project_root / "splits" / dataset / protocol / f"seed_{seed}.npz"
    if split_path.exists():
        print(f"SPLIT existing: {split_path}")
        return split_path

    command = [
        sys.executable,
        str(project_root / "scripts" / "make_split.py"),
        "--data-path",
        str(data_path),
        "--gt-path",
        str(gt_path),
        "--data-key",
        spec["data_key"],
        "--gt-key",
        spec["gt_key"],
        "--protocol",
        protocol,
        "--seed",
        str(seed),
        "--train-per-class",
        str(args.train_per_class),
        "--val-per-class",
        str(args.val_per_class),
        "--output",
        str(split_path),
    ]
    if protocol == "spatial_block":
        command.extend(
            [
                "--min-test-per-class",
                str(args.min_test_per_class),
                "--block-size",
                str(args.block_size),
                "--buffer-pixels",
                str(args.buffer_pixels),
                "--trials",
                str(args.trials),
            ]
        )
    subprocess.run(command, cwd=project_root, check=True)
    return split_path


def _base_config(
    *,
    project_root: Path,
    data_path: Path,
    gt_path: Path,
    spec: dict[str, str],
    dataset: str,
    experiment: str,
    args: argparse.Namespace,
) -> dict:
    return {
        "seed": 0,
        "data": {
            "name": dataset,
            "data_path": str(data_path),
            "gt_path": str(gt_path),
            "data_key": spec["data_key"],
            "gt_key": spec["gt_key"],
            "zscore_clip": 8.0,
        },
        "protocol": {"name": "spatial_block", "split_path": ""},
        "model": {
            "name": "",
            "hidden_dim": args.hidden_dim,
            "spectral": "conv1d",
            "fusion": "reliability",
            "dropout": 0.1,
        },
        "training": {
            "patch_size": args.patch_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "aux_weight": 0.2,
            "amp": True,
            "deterministic": True,
        },
        "output": {
            "root": str(project_root / "results"),
            "experiment": experiment,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable, immutable KMFM experiment matrix in Colab"
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="pavia_university",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--experiment", default="pilot_v1")
    parser.add_argument(
        "--suite",
        choices=("legacy", "calibrated_v3", "oasd_v6"),
        default="legacy",
        help=(
            "Model matrix to run; calibrated_v3 is the six-model mechanism test and "
            "oasd_v6 is the fixed spatial-only distillation development matrix."
        ),
    )
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0,1"))
    parser.add_argument(
        "--protocols",
        type=_csv_values,
        default=_csv_values("random_pixel,spatial_block"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--train-per-class", type=int, default=30)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--min-test-per-class", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--buffer-pixels", type=int, default=3)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument(
        "--variants",
        type=_csv_values,
        help="Optional comma-separated model names; defaults to all core variants",
    )
    parser.add_argument(
        "--recover-incomplete",
        action="store_true",
        help=(
            "Archive each non-successful run under results/_incomplete and rerun it. "
            "Successful immutable runs are still skipped."
        ),
    )
    parser.add_argument("--no-aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if not (project_root / "src" / "kmfm").is_dir():
        raise SystemExit(f"KMFM source tree not found: {project_root}")
    if any(protocol not in {"random_pixel", "spatial_block"} for protocol in args.protocols):
        raise SystemExit("--protocols only accepts random_pixel and spatial_block")

    sys.path.insert(0, str(project_root / "src"))
    from kmfm.engine import resolve_run_dir, run_experiment

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.suite == "calibrated_v3":
        available_variants = CALIBRATED_V3_VARIANTS
    elif args.suite == "oasd_v6":
        available_variants = OASD_V6_VARIANTS
    else:
        available_variants = CORE_VARIANTS + (
            ABLATION_VARIANTS if args.include_ablations else []
        )
    variants_by_name = {variant["name"]: variant for variant in available_variants}
    if args.variants:
        unknown = [name for name in args.variants if name not in variants_by_name]
        if unknown:
            raise SystemExit(f"Unknown or disabled --variants: {', '.join(unknown)}")
        variants = [variants_by_name[name] for name in args.variants]
    else:
        variants = available_variants

    for dataset in datasets:
        spec = DATASETS[dataset]
        data_path = data_root / spec["data"]
        gt_path = data_root / spec["gt"]
        if not data_path.is_file() or not gt_path.is_file():
            raise SystemExit(f"Dataset files not found: {data_path} / {gt_path}")
        base = _base_config(
            project_root=project_root,
            data_path=data_path,
            gt_path=gt_path,
            spec=spec,
            dataset=dataset,
            experiment=args.experiment,
            args=args,
        )
        for protocol in args.protocols:
            for seed in args.seeds:
                split_path = _make_split(
                    project_root=project_root,
                    data_path=data_path,
                    gt_path=gt_path,
                    spec=spec,
                    dataset=dataset,
                    protocol=protocol,
                    seed=seed,
                    args=args,
                )
                for variant in variants:
                    config = copy.deepcopy(base)
                    config["seed"] = seed
                    config["protocol"] = {
                        "name": protocol,
                        "split_path": str(split_path),
                    }
                    config["model"].update(variant)
                    run_dir = resolve_run_dir(config)
                    status = _read_status(run_dir)
                    if status == "success":
                        print(f"SKIP successful: {run_dir}")
                        continue
                    if run_dir.exists() and any(run_dir.iterdir()):
                        if not args.recover_incomplete:
                            raise RuntimeError(
                                f"Incomplete immutable run exists ({status}): {run_dir}. "
                                "Inspect status.json, then use a new --experiment name or "
                                "explicitly pass --recover-incomplete."
                            )
                        archive_dir = _archive_incomplete_run(
                            run_dir,
                            results_root=project_root / "results",
                            experiment=args.experiment,
                            status=status,
                        )
                        print(
                            f"ARCHIVE incomplete ({status}): {run_dir} -> {archive_dir}"
                        )
                    print(f"RUN {dataset} {protocol} {variant['name']} seed={seed}")
                    run_experiment(config)

    if not args.no_aggregate:
        report_dir = project_root / "reports" / args.experiment
        if args.suite == "calibrated_v3":
            reference_models = (
                "lassf_mlp_spatial_only_v3_h64,lassf_mlp_gate_norm_v3_h64"
            )
        elif args.suite == "oasd_v6":
            reference_models = (
                "lassf_mlp_spatial_only_v6_h64,lassf_mlp_uniform_distill_v6_h64"
            )
        else:
            reference_models = "lassf_conv1d_concat_h64"
        subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "aggregate.py"),
                "--results-root",
                str(project_root / "results" / args.experiment),
                "--output-dir",
                str(report_dir),
                "--reference-model",
                reference_models,
            ],
            cwd=project_root,
            check=True,
        )
        if args.suite == "calibrated_v3":
            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "evaluate_mechanism.py"),
                    "--per-run",
                    str(report_dir / "per_run.csv"),
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=project_root,
                check=True,
            )
        elif args.suite == "oasd_v6":
            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "evaluate_oasd.py"),
                    "--per-run",
                    str(report_dir / "per_run.csv"),
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=project_root,
                check=True,
            )
        print(f"REPORT {report_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
