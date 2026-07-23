from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kmfm.artifacts import build_manifest, environment_snapshot, json_dump, save_confusion_csv
from kmfm.data import load_hsi, standardize_cube
from kmfm.engine import _make_loaders, _run_loader, _spearman, set_reproducibility
from kmfm.logit_fusion import ADLFPolicy, apply_adlf_policy, fit_adlf_policy
from kmfm.metrics import classification_metrics, probabilistic_metrics, routing_diagnostics
from kmfm.model import LASSFNet
from kmfm.splits import TRAIN, load_split


PROJECT_DEFAULT = Path("/content/drive/MyDrive/Colab/Unsupervised/KMFM")
VARIANTS = (
    "replay_spatial_logit_v4",
    "replay_spectral_logit_v4",
    "replay_mean_logit_v4",
    "replay_global_logit_v4",
    "replay_adlf_v4",
)


def _seeds(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return values


def _status(run_dir: Path) -> str | None:
    path = run_dir / "status.json"
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("state"))
    except (OSError, json.JSONDecodeError):
        return "invalid"


def _archive_incomplete(run_dir: Path, results_root: Path, experiment: str) -> Path:
    relative = run_dir.relative_to(results_root / experiment)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = results_root / "_incomplete" / experiment / relative.parent / f"{relative.name}__{stamp}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    run_dir.rename(archive)
    return archive


def _load_source_outputs(source_run: Path) -> tuple[dict, dict, dict, float, int]:
    status = _status(source_run)
    if status != "success":
        raise RuntimeError(f"Source run is not successful ({status}): {source_run}")
    config = json.loads((source_run / "resolved_config.json").read_text(encoding="utf-8"))
    checkpoint_path = source_run / "checkpoint_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")

    seed = int(config["seed"])
    set_reproducibility(seed, True)
    data_cfg = config["data"]
    hsi = load_hsi(
        data_cfg["data_path"],
        data_cfg["gt_path"],
        data_cfg.get("data_key"),
        data_cfg.get("gt_key"),
    )
    split_path = Path(config["protocol"]["split_path"])
    split = load_split(split_path)
    fit_mask = split.region_map == TRAIN if split.metadata["protocol"] == "spatial_block" else split.train_mask
    cube, _ = standardize_cube(hsi.cube, fit_mask, clip=data_cfg.get("zscore_clip", 8.0))
    _, val_loader, test_loader = _make_loaders(cube, hsi.labels, split, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = LASSFNet(
        bands=cube.shape[-1],
        num_classes=hsi.num_classes,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        spectral=model_cfg.get("spectral", "conv1d"),
        fusion=model_cfg.get("fusion", "reliability"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        normalize_branches=bool(model_cfg.get("normalize_branches", False)),
        entropy_temperature=float(model_cfg.get("entropy_temperature", 0.25)),
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_branch_temperatures(1.0, 1.0)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    aux_weight = float(config["training"].get("aux_weight", 0.2))
    val_result = _run_loader(
        model, val_loader, device, hsi.num_classes, criterion, None, None, aux_weight, False
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    test_result = _run_loader(
        model, test_loader, device, hsi.num_classes, criterion, None, None, aux_weight, False
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return config, val_result, test_result, inference_seconds, parameter_count


def _variant_metrics(
    *,
    variant: str,
    seed: int,
    dataset: str,
    protocol: str,
    logits: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    global_predictions: np.ndarray,
    policy: ADLFPolicy,
    inference_seconds: float,
    parameter_count: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predictions = np.asarray(logits).argmax(axis=1)
    spatial_predictions = spatial_logits.argmax(axis=1)
    spectral_predictions = spectral_logits.argmax(axis=1)
    metrics, confusion = classification_metrics(targets, predictions, logits.shape[1])
    metrics.update(probabilistic_metrics(targets, logits))
    metrics.update(
        routing_diagnostics(
            targets, predictions, spatial_predictions, spectral_predictions, weights
        )
    )
    spatial_probability = probabilistic_metrics(targets, spatial_logits)
    spectral_probability = probabilistic_metrics(targets, spectral_logits)
    metrics.update(
        {f"spatial_branch_{name}": value for name, value in spatial_probability.items()}
    )
    metrics.update(
        {f"spectral_branch_{name}": value for name, value in spectral_probability.items()}
    )

    disagreement = spatial_predictions != spectral_predictions
    improved = (~(global_predictions == targets)) & (predictions == targets)
    harmed = (global_predictions == targets) & (~(predictions == targets))
    radius = policy.residual_radius if variant == "replay_adlf_v4" else 0.0
    metrics.update(
        {
            "dataset": dataset,
            "protocol": protocol,
            "model": variant,
            "seed": seed,
            "parameter_count": parameter_count,
            "training_seconds": 0.0,
            "test_inference_seconds": inference_seconds,
            "gate_mean": float(np.mean(weights)),
            "gate_std": float(np.std(weights)),
            "contribution_ratio_mean": float(np.mean(weights)),
            "contribution_ratio_std": float(np.std(weights)),
            "gate_entropy_gap_spearman": _spearman(
                weights,
                probabilistic_metrics_per_sample_entropy(spectral_logits)
                - probabilistic_metrics_per_sample_entropy(spatial_logits),
            ),
            "spatial_temperature": policy.spatial_temperature,
            "spectral_temperature": policy.spectral_temperature,
            "selected_alpha": _variant_alpha(variant, policy),
            "selected_radius": radius,
            "validation_nll": policy.validation_nll[_variant_key(variant)],
            "prediction_disagreement_count": int(disagreement.sum()),
            "prediction_disagreement_fraction": float(disagreement.mean()),
            "vs_global_improved_count": int(improved.sum()),
            "vs_global_harmed_count": int(harmed.sum()),
            "vs_global_net_corrected": int(improved.sum() - harmed.sum()),
        }
    )
    return metrics, confusion, predictions


def probabilistic_metrics_per_sample_entropy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -(probabilities * np.log(probabilities.clip(1e-12, 1.0))).sum(axis=1)


def _variant_key(variant: str) -> str:
    return {
        "replay_spatial_logit_v4": "spatial",
        "replay_spectral_logit_v4": "spectral",
        "replay_mean_logit_v4": "mean",
        "replay_global_logit_v4": "global",
        "replay_adlf_v4": "adlf",
    }[variant]


def _variant_alpha(variant: str, policy: ADLFPolicy) -> float:
    return {
        "replay_spatial_logit_v4": 1.0,
        "replay_spectral_logit_v4": 0.0,
        "replay_mean_logit_v4": 0.5,
        "replay_global_logit_v4": policy.global_alpha,
        "replay_adlf_v4": policy.global_alpha,
    }[variant]


def _save_variant(
    *,
    run_dir: Path,
    metrics: dict[str, Any],
    confusion: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    policy: ADLFPolicy,
    source_run: Path,
    source_config: dict,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(run_dir / "status.json", {"state": "running", "seed": metrics["seed"]})
    json_dump(run_dir / "policy.json", policy.to_dict())
    json_dump(
        run_dir / "source.json",
        {"source_run": str(source_run), "source_model": source_config["model"]["name"]},
    )
    np.save(run_dir / "test_targets.npy", np.asarray(targets, dtype=np.int16))
    np.save(run_dir / "test_predictions.npy", np.asarray(predictions, dtype=np.int16))
    np.save(run_dir / "test_logits.npy", np.asarray(logits, dtype=np.float32))
    np.save(run_dir / "spatial_weights.npy", np.asarray(weights, dtype=np.float32))
    for name, values in (extra_arrays or {}).items():
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError(f"Invalid extra array name: {name}")
        np.save(run_dir / name, np.asarray(values))
    np.save(run_dir / "confusion_matrix.npy", confusion)
    save_confusion_csv(run_dir / "confusion_matrix.csv", confusion)
    json_dump(run_dir / "metrics.json", metrics)
    json_dump(run_dir / "environment.json", environment_snapshot())
    json_dump(run_dir / "status.json", {"state": "success", "seed": metrics["seed"]})

    checkpoint_path = source_run / "checkpoint_best.pt"
    split_path = Path(source_config["protocol"]["split_path"])
    data_cfg = source_config["data"]
    output_files = [path for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = build_manifest(
        run_dir,
        input_files={
            "source_checkpoint": checkpoint_path,
            "source_config": source_run / "resolved_config.json",
            "data": data_cfg["data_path"],
            "ground_truth": data_cfg["gt_path"],
            "split": split_path,
        },
        output_files=output_files,
        extra={
            "status": "success",
            "seed": metrics["seed"],
            "dataset": metrics["dataset"],
            "protocol": metrics["protocol"],
            "model": metrics["model"],
            "source_run": str(source_run),
        },
    )
    json_dump(run_dir / "manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay calibrated branch logits with global and bounded disagreement fusion"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--source-experiment", default="pavia_calibrated_v3")
    parser.add_argument("--output-experiment", default="pavia_adlf_replay_v4")
    parser.add_argument("--dataset", default="pavia_university")
    parser.add_argument("--protocol", default="spatial_block")
    parser.add_argument("--source-model", default="lassf_mlp_concat_norm_v3_h64")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0,1,2,3,4"))
    parser.add_argument("--recover-incomplete", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = project_root / "results"
    output_root = results_root / args.output_experiment

    for seed in args.seeds:
        run_dirs = {
            variant: output_root / args.dataset / args.protocol / variant / f"seed_{seed}"
            for variant in VARIANTS
        }
        pending: list[str] = []
        for variant, run_dir in run_dirs.items():
            status = _status(run_dir)
            if status == "success":
                print(f"SKIP successful: {run_dir}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                if not args.recover_incomplete:
                    raise RuntimeError(
                        f"Incomplete immutable replay exists ({status}): {run_dir}. "
                        "Inspect it or pass --recover-incomplete."
                    )
                archive = _archive_incomplete(run_dir, results_root, args.output_experiment)
                print(f"ARCHIVE incomplete ({status}): {run_dir} -> {archive}")
            pending.append(variant)
        if not pending:
            continue

        source_run = (
            results_root
            / args.source_experiment
            / args.dataset
            / args.protocol
            / args.source_model
            / f"seed_{seed}"
        )
        print(f"REPLAY {args.dataset} {args.protocol} {args.source_model} seed={seed}")
        source_config, val_result, test_result, inference_seconds, parameter_count = (
            _load_source_outputs(source_run)
        )
        policy = fit_adlf_policy(
            val_result["spatial_logits"],
            val_result["spectral_logits"],
            val_result["targets"],
        )
        test_variants = apply_adlf_policy(
            policy, test_result["spatial_logits"], test_result["spectral_logits"]
        )
        spatial_calibrated = test_result["spatial_logits"] / policy.spatial_temperature
        spectral_calibrated = test_result["spectral_logits"] / policy.spectral_temperature
        global_predictions = test_variants["replay_global_logit_v4"][0].argmax(axis=1)

        for variant in pending:
            logits, weights = test_variants[variant]
            metrics, confusion, predictions = _variant_metrics(
                variant=variant,
                seed=seed,
                dataset=args.dataset,
                protocol=args.protocol,
                logits=logits,
                weights=weights,
                targets=test_result["targets"],
                spatial_logits=spatial_calibrated,
                spectral_logits=spectral_calibrated,
                global_predictions=global_predictions,
                policy=policy,
                inference_seconds=inference_seconds,
                parameter_count=parameter_count,
            )
            _save_variant(
                run_dir=run_dirs[variant],
                metrics=metrics,
                confusion=confusion,
                predictions=predictions,
                logits=logits,
                weights=weights,
                targets=test_result["targets"],
                policy=policy,
                source_run=source_run,
                source_config=source_config,
            )
            print(f"SAVED {variant} seed={seed} oa={metrics['oa']:.6f}")

    if args.no_aggregate:
        return
    report_dir = project_root / "reports" / args.output_experiment
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "aggregate.py"),
            "--results-root",
            str(output_root),
            "--output-dir",
            str(report_dir),
            "--reference-model",
            "replay_spatial_logit_v4,replay_global_logit_v4",
        ],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_adlf_replay.py"),
            "--per-run",
            str(report_dir / "per_run.csv"),
            "--output-dir",
            str(report_dir),
        ],
        cwd=project_root,
        check=True,
    )
    print(f"REPORT {report_dir / 'adlf_replay_decision.md'}")


if __name__ == "__main__":
    main()
