from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_status(run_dir: Path) -> str | None:
    path = run_dir / "status.json"
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("state"))
    except (OSError, ValueError, TypeError):
        return "invalid"


def _archive_incomplete(run_dir: Path, audit_root: Path) -> Path:
    relative = run_dir.relative_to(audit_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = audit_root / "_incomplete" / relative.parent / f"{relative.name}__{timestamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    return destination


def _train_primary_model(
    *,
    train_dataset: Any,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    bands: int,
    num_classes: int,
    encoder: str,
    fusion: str,
    hidden_dim: int,
    dropout: float,
    normalize_branches: bool,
    batch_size: int,
    num_workers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    aux_weight: float,
    amp_enabled: bool,
    deterministic: bool,
    seed: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[np.ndarray, float]:
    from kmfm.engine import (
        _build_model,
        _make_scaler,
        _run_loader,
        _subset_loader,
        set_reproducibility,
    )

    set_reproducibility(seed, deterministic)
    train_loader = _subset_loader(
        train_dataset,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
    )
    holdout_loader = _subset_loader(
        train_dataset,
        holdout_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
    )
    model = _build_model(
        bands=bands,
        num_classes=num_classes,
        model_cfg={
            "hidden_dim": hidden_dim,
            "spectral": encoder,
            "fusion": fusion,
            "dropout": dropout,
            "normalize_branches": normalize_branches,
        },
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    scaler = _make_scaler(device, amp_enabled)
    started = time.perf_counter()
    for _ in range(epochs):
        _run_loader(
            model,
            train_loader,
            device,
            num_classes,
            criterion,
            optimizer,
            scaler,
            aux_weight,
            amp_enabled,
        )
        scheduler.step()
    holdout = _run_loader(
        model,
        holdout_loader,
        device,
        num_classes,
        criterion,
        None,
        None,
        aux_weight,
        amp_enabled,
    )
    elapsed = time.perf_counter() - started
    predictions = np.asarray(holdout["predictions"], dtype=np.int64)
    del model, optimizer, scheduler, scaler, train_loader, holdout_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, elapsed


def _audit_encoder_seed(
    *,
    train_dataset: Any,
    targets: np.ndarray,
    coords: np.ndarray,
    bands: int,
    num_classes: int,
    encoder: str,
    seed: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    from kmfm.distillation import stratified_folds
    from kmfm.spectral_audit import StabilityThresholds, repeated_oof_stability_profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp) and device.type == "cuda"
    criterion = nn.CrossEntropyLoss()
    spatial_predictions = np.full((args.repeats, len(targets)), -1, dtype=np.int64)
    spectral_predictions = np.full((args.repeats, len(targets)), -1, dtype=np.int64)
    all_indices = np.arange(len(targets), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for repeat in range(args.repeats):
        folds = stratified_folds(
            targets,
            n_splits=args.folds,
            seed=seed + 1701 + repeat * 10007,
        )
        for fold_id, holdout_indices in enumerate(folds):
            train_indices = np.setdiff1d(all_indices, holdout_indices, assume_unique=True)
            fold_seed = seed * 100003 + repeat * 1009 + fold_id + 1
            spatial_fold, spatial_seconds = _train_primary_model(
                train_dataset=train_dataset,
                train_indices=train_indices,
                holdout_indices=holdout_indices,
                bands=bands,
                num_classes=num_classes,
                encoder=encoder,
                fusion="spatial_only",
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                normalize_branches=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                aux_weight=args.aux_weight,
                amp_enabled=amp_enabled,
                deterministic=args.deterministic,
                seed=fold_seed,
                device=device,
                criterion=criterion,
            )
            spectral_fold, spectral_seconds = _train_primary_model(
                train_dataset=train_dataset,
                train_indices=train_indices,
                holdout_indices=holdout_indices,
                bands=bands,
                num_classes=num_classes,
                encoder=encoder,
                fusion="spectral_only",
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                normalize_branches=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                aux_weight=args.aux_weight,
                amp_enabled=amp_enabled,
                deterministic=args.deterministic,
                seed=fold_seed,
                device=device,
                criterion=criterion,
            )
            spatial_predictions[repeat, holdout_indices] = spatial_fold
            spectral_predictions[repeat, holdout_indices] = spectral_fold
            fold_targets = targets[holdout_indices]
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold_id,
                    "train_count": int(len(train_indices)),
                    "holdout_count": int(len(holdout_indices)),
                    "spatial_oa": float(np.mean(spatial_fold == fold_targets)),
                    "spectral_oa": float(np.mean(spectral_fold == fold_targets)),
                    "spatial_training_seconds": float(spatial_seconds),
                    "spectral_training_seconds": float(spectral_seconds),
                }
            )
            print(
                f"AUDIT encoder={encoder} seed={seed} repeat={repeat} fold={fold_id} "
                f"spatial={fold_rows[-1]['spatial_oa']:.3f} "
                f"spectral={fold_rows[-1]['spectral_oa']:.3f}",
                flush=True,
            )

    if np.any(spatial_predictions < 0) or np.any(spectral_predictions < 0):
        raise RuntimeError("Repeated OOF prediction arrays were not filled completely")
    thresholds = StabilityThresholds()
    profile = repeated_oof_stability_profile(
        targets,
        spatial_predictions,
        spectral_predictions,
        num_classes,
        thresholds=thresholds,
    )
    profile.update(
        {
            "dataset": args.dataset,
            "protocol": args.protocol,
            "encoder": encoder,
            "seed": seed,
            "folds": args.folds,
            "epochs": args.epochs,
            "elapsed_seconds": float(time.perf_counter() - started),
            "fold_results": fold_rows,
        }
    )
    np.savez_compressed(
        run_dir / "oof_predictions.npz",
        targets=targets.astype(np.int16),
        coords=coords.astype(np.int32),
        spatial_predictions=spatial_predictions.astype(np.int16),
        spectral_predictions=spectral_predictions.astype(np.int16),
    )
    (run_dir / "audit.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    return profile


def _decision_markdown(decision: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Spectral Teacher V7 OOF Stability Audit",
        "",
        "> Training-region repeated OOF diagnostic. Test labels are not loaded or used.",
        "",
        f"- Dataset/protocol: `{args.dataset}/{args.protocol}`",
        f"- Repeats/folds/epochs: `{args.repeats}/{args.folds}/{args.epochs}`",
        f"- Decision: **{decision['decision']}**",
        f"- Selected encoder: `{decision['selected_encoder']}`",
        f"- Diagnostic best encoder: `{decision['diagnostic_best_encoder']}`",
        "",
        "| Encoder | Pass | Qualified seeds | Stable classes (median) | OOF OA gap | Stable net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in decision["evaluations"]:
        lines.append(
            f"| {row['encoder']} | {row['passed']} | "
            f"{row['qualified_seed_count']}/{row['seed_count']} | "
            f"{row['stable_class_count_median']:.2f} | "
            f"{100.0 * row['global_oa_gap_mean']:.3f} pp | "
            f"{row['stable_net_correct_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered Gate",
            "",
            "- A class is stable only when spectral advantage is positive in at least 2/3 repeats, mean advantage is at least 5 percentage points, and the worst repeat is no worse than -5 points.",
            "- A seed passes with at least three stable classes and positive stable instance-level net corrections.",
            "- An encoder passes with at least 80% passing seeds, positive mean stable net corrections, and mean global OOF OA no more than 5 points below spatial.",
            "- `DEVELOPMENT_GO` unlocks instance-level residual distillation. `DEVELOPMENT_NO_GO` terminates this teacher-distillation branch.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated training-region OOF audit for v7 spectral teachers"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="pavia_university")
    parser.add_argument("--protocol", choices=("random_pixel", "spatial_block"), default="spatial_block")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--experiment", default="pavia_spectral_teacher_audit_v7")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0,1,2,3,4"))
    parser.add_argument("--encoders", type=_csv_values, default=_csv_values("mlp,conv1d"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--recover-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 2 or args.folds < 2 or args.epochs < 1:
        raise SystemExit("--repeats and --folds must be at least 2; --epochs must be positive")
    if any(encoder not in {"mlp", "conv1d"} for encoder in args.encoders):
        raise SystemExit("--encoders only accepts mlp and conv1d")
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    sys.path.insert(0, str(project_root / "src"))

    from kmfm.data import HSIPatchDataset, load_hsi, standardize_cube
    from kmfm.spectral_audit import evaluate_encoder_profiles, select_encoder
    from kmfm.splits import TRAIN, load_split

    spec = DATASETS[args.dataset]
    data_path = data_root / spec["data"]
    gt_path = data_root / spec["gt"]
    if not data_path.is_file() or not gt_path.is_file():
        raise SystemExit(f"Dataset files not found: {data_path} / {gt_path}")
    hsi = load_hsi(data_path, gt_path, spec["data_key"], spec["gt_key"])
    audit_root = project_root / "audits"
    experiment_root = audit_root / args.experiment / args.dataset / args.protocol
    profiles_by_encoder: dict[str, list[dict[str, Any]]] = {
        encoder: [] for encoder in args.encoders
    }

    for seed in args.seeds:
        pending = []
        for encoder in args.encoders:
            run_dir = experiment_root / encoder / f"seed_{seed}"
            status = _read_status(run_dir)
            if status == "success":
                print(f"SKIP successful: {run_dir}")
                profiles_by_encoder[encoder].append(
                    json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
                )
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                if not args.recover_incomplete:
                    raise RuntimeError(
                        f"Incomplete immutable audit exists ({status}): {run_dir}. "
                        "Inspect status.json or pass --recover-incomplete."
                    )
                archived = _archive_incomplete(run_dir, audit_root)
                print(f"ARCHIVE incomplete ({status}): {run_dir} -> {archived}")
            pending.append((encoder, run_dir))
        if not pending:
            continue

        split_path = project_root / "splits" / args.dataset / args.protocol / f"seed_{seed}.npz"
        split = load_split(split_path)
        if int(split.metadata["seed"]) != seed:
            raise ValueError(f"Split seed mismatch: {split_path}")
        fit_mask = split.region_map == TRAIN if args.protocol == "spatial_block" else split.train_mask
        cube, _ = standardize_cube(hsi.cube, fit_mask, clip=8.0)
        train_dataset = HSIPatchDataset(
            cube=cube,
            labels=hsi.labels,
            center_mask=split.train_mask,
            patch_size=args.patch_size,
            region_map=split.region_map,
            region_value=TRAIN,
            allow_full_context=bool(split.metadata.get("allow_full_context", False)),
        )
        targets = np.asarray(
            [train_dataset.labels[row, col] - 1 for row, col in train_dataset.coords],
            dtype=np.int64,
        )
        for encoder, run_dir in pending:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "seed": seed, "encoder": encoder}, indent=2),
                encoding="utf-8",
            )
            try:
                profile = _audit_encoder_seed(
                    train_dataset=train_dataset,
                    targets=targets,
                    coords=train_dataset.coords,
                    bands=cube.shape[-1],
                    num_classes=hsi.num_classes,
                    encoder=encoder,
                    seed=seed,
                    args=args,
                    run_dir=run_dir,
                )
                profiles_by_encoder[encoder].append(profile)
                (run_dir / "status.json").write_text(
                    json.dumps({"state": "success", "seed": seed, "encoder": encoder}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                (run_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "state": "failed",
                            "seed": seed,
                            "encoder": encoder,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise

    evaluations = [
        evaluate_encoder_profiles(encoder, profiles_by_encoder[encoder])
        for encoder in args.encoders
    ]
    decision = select_encoder(evaluations)
    report_dir = project_root / "reports" / args.experiment
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "spectral_teacher_audit_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = _decision_markdown(decision, args)
    (report_dir / "spectral_teacher_audit_decision.md").write_text(
        markdown, encoding="utf-8"
    )

    per_seed_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for encoder, profiles in profiles_by_encoder.items():
        for profile in profiles:
            per_seed_rows.append(
                {
                    "dataset": args.dataset,
                    "protocol": args.protocol,
                    "encoder": encoder,
                    "seed": profile["seed"],
                    "spatial_oa_mean": profile["spatial_oa_mean"],
                    "spectral_oa_mean": profile["spectral_oa_mean"],
                    "global_oa_gap": profile["global_oa_gap"],
                    "stable_class_count": profile["stable_class_count"],
                    "stable_beneficial_count": profile["stable_beneficial_count"],
                    "stable_harmful_count": profile["stable_harmful_count"],
                    "stable_net_correct": profile["stable_net_correct"],
                    "seed_qualified": profile["seed_qualified"],
                    "elapsed_seconds": profile["elapsed_seconds"],
                }
            )
            for row in profile["per_class"]:
                per_class_rows.append(
                    {
                        "dataset": args.dataset,
                        "protocol": args.protocol,
                        "encoder": encoder,
                        "seed": profile["seed"],
                        "class_id": row["class_id"],
                        "count": row["count"],
                        "positive_repeats": row["positive_repeats"],
                        "mean_advantage": row["mean_advantage"],
                        "worst_advantage": row["worst_advantage"],
                        "stable_beneficial_count": row["stable_beneficial_count"],
                        "stable_harmful_count": row["stable_harmful_count"],
                        "stable_net_correct": row["stable_net_correct"],
                        "stable_positive": row["stable_positive"],
                    }
                )
    _write_csv(report_dir / "per_seed.csv", per_seed_rows)
    _write_csv(report_dir / "per_class.csv", per_class_rows)
    print(markdown, flush=True)
    print(f"REPORT {report_dir}", flush=True)


if __name__ == "__main__":
    main()
