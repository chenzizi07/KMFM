from __future__ import annotations

import copy
import csv
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import build_manifest, environment_snapshot, json_dump, save_confusion_csv
from .data import HSIPatchDataset, load_hsi, standardize_cube
from .metrics import classification_metrics
from .model import LASSFNet
from .splits import TEST, TRAIN, VAL, load_split


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_name(config: dict[str, Any]) -> str:
    model = config["model"]
    return model.get(
        "name",
        f"lassf_{model.get('spectral', 'conv1d')}_{model.get('fusion', 'reliability')}_h{model.get('hidden_dim', 64)}",
    )


def resolve_run_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["output"]["root"])
    experiment = config["output"]["experiment"]
    return (
        output_root
        / experiment
        / config["data"]["name"]
        / config["protocol"]["name"]
        / _model_name(config)
        / f"seed_{int(config['seed'])}"
    )


def _make_loaders(
    cube: np.ndarray,
    labels: np.ndarray,
    split: Any,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    training = config["training"]
    patch_size = int(training.get("patch_size", 11))
    allow_full = bool(split.metadata.get("allow_full_context", False))
    common = {
        "cube": cube,
        "labels": labels,
        "patch_size": patch_size,
        "region_map": split.region_map,
        "allow_full_context": allow_full,
    }
    train_dataset = HSIPatchDataset(
        center_mask=split.train_mask, region_value=TRAIN, **common
    )
    val_dataset = HSIPatchDataset(center_mask=split.val_mask, region_value=VAL, **common)
    test_dataset = HSIPatchDataset(center_mask=split.test_mask, region_value=TEST, **common)
    batch_size = int(training.get("batch_size", 64))
    workers = int(training.get("num_workers", 0))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def _run_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    aux_weight: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    coords_all: list[np.ndarray] = []
    gates_all: list[np.ndarray] = []
    spatial_entropy_all: list[np.ndarray] = []
    spectral_entropy_all: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for patch, target, coords, context_mask in loader:
            patch = patch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            context_mask = context_mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(patch, context_mask=context_mask)
                loss = criterion(output["logits"], target)
                loss = loss + aux_weight * (
                    criterion(output["spatial_logits"], target)
                    + criterion(output["spectral_logits"], target)
                )
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            targets_all.append(target.detach().cpu().numpy())
            predictions_all.append(output["logits"].argmax(dim=-1).detach().cpu().numpy())
            coords_all.append(coords.numpy())
            gates_all.append(output["gate"].detach().float().cpu().numpy())
            spatial_entropy_all.append(output["spatial_entropy"].detach().float().cpu().numpy())
            spectral_entropy_all.append(output["spectral_entropy"].detach().float().cpu().numpy())

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    metrics, confusion = classification_metrics(targets, predictions, num_classes)
    return {
        "loss": float(np.mean(losses)),
        "metrics": metrics,
        "confusion": confusion,
        "targets": targets,
        "predictions": predictions,
        "coords": np.concatenate(coords_all),
        "gate": np.concatenate(gates_all),
        "spatial_entropy": np.concatenate(spatial_entropy_all),
        "spectral_entropy": np.concatenate(spectral_entropy_all),
    }


def _write_curves(path: Path, curves: list[dict[str, Any]]) -> None:
    if not curves:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0].keys()))
        writer.writeheader()
        writer.writerows(curves)


def run_experiment(config: dict[str, Any]) -> Path:
    config = copy.deepcopy(config)
    seed = int(config["seed"])
    set_reproducibility(seed, bool(config["training"].get("deterministic", True)))
    run_dir = resolve_run_dir(config)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a new output.experiment name; "
            "successful runs are immutable."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dump(run_dir / "status.json", {"state": "running", "seed": seed})
    json_dump(run_dir / "resolved_config.json", config)

    try:
        data_cfg = config["data"]
        hsi = load_hsi(
            data_cfg["data_path"],
            data_cfg["gt_path"],
            data_cfg.get("data_key"),
            data_cfg.get("gt_key"),
        )
        split_path = Path(config["protocol"]["split_path"])
        split = load_split(split_path)
        if split.train_mask.shape != hsi.labels.shape:
            raise ValueError(
                f"Split shape {split.train_mask.shape} does not match labels {hsi.labels.shape}"
            )
        if int(split.metadata["seed"]) != seed:
            raise ValueError(
                f"Config seed {seed} does not match split seed {split.metadata['seed']}"
            )
        protocol_name = str(split.metadata["protocol"])
        if config["protocol"]["name"] != protocol_name:
            raise ValueError(
                f"Config protocol {config['protocol']['name']} != split protocol {protocol_name}"
            )

        if protocol_name == "spatial_block":
            fit_mask = split.region_map == TRAIN
        else:
            fit_mask = split.train_mask
        cube, preprocessing = standardize_cube(
            hsi.cube, fit_mask, clip=config["data"].get("zscore_clip", 8.0)
        )
        json_dump(
            run_dir / "data_manifest.json",
            {
                "cube_shape": list(cube.shape),
                "label_shape": list(hsi.labels.shape),
                "num_classes": hsi.num_classes,
                "class_mapping": hsi.class_mapping,
                "selected_data_key": hsi.data_key,
                "selected_gt_key": hsi.gt_key,
                "preprocessing": preprocessing,
                "split_metadata": split.metadata,
            },
        )
        train_loader, val_loader, test_loader = _make_loaders(cube, hsi.labels, split, config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config["model"]
        model = LASSFNet(
            bands=cube.shape[-1],
            num_classes=hsi.num_classes,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            spectral=model_cfg.get("spectral", "conv1d"),
            fusion=model_cfg.get("fusion", "reliability"),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(device)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        criterion = nn.CrossEntropyLoss()
        training_cfg = config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("lr", 3e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(training_cfg.get("epochs", 200))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
        if amp_enabled:
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            scaler = None
        aux_weight = float(training_cfg.get("aux_weight", 0.2))
        patience = int(training_cfg.get("patience", 30))

        best_oa = -float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        curves: list[dict[str, Any]] = []
        _sync(device)
        train_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            train_result = _run_loader(
                model,
                train_loader,
                device,
                hsi.num_classes,
                criterion,
                optimizer,
                scaler,
                aux_weight,
                amp_enabled,
            )
            val_result = _run_loader(
                model,
                val_loader,
                device,
                hsi.num_classes,
                criterion,
                None,
                None,
                aux_weight,
                amp_enabled,
            )
            scheduler.step()
            val_oa = float(val_result["metrics"]["oa"])
            curves.append(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_result["loss"],
                    "train_oa": train_result["metrics"]["oa"],
                    "val_loss": val_result["loss"],
                    "val_oa": val_oa,
                    "val_aa": val_result["metrics"]["aa"],
                    "val_kappa": val_result["metrics"]["kappa"],
                }
            )
            if val_oa > best_oa + 1e-12:
                best_oa = val_oa
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_oa": val_oa,
                        "model_config": model_cfg,
                        "bands": cube.shape[-1],
                        "num_classes": hsi.num_classes,
                    },
                    run_dir / "checkpoint_best.pt",
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        _sync(device)
        training_seconds = time.perf_counter() - train_started
        _write_curves(run_dir / "curves.csv", curves)

        try:
            checkpoint = torch.load(
                run_dir / "checkpoint_best.pt", map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        # Warm-up one validation batch before measuring final test inference.
        warm_patch, _, _, warm_mask = next(iter(val_loader))
        with torch.no_grad():
            model(warm_patch.to(device), context_mask=warm_mask.to(device))
        _sync(device)
        test_started = time.perf_counter()
        test_result = _run_loader(
            model,
            test_loader,
            device,
            hsi.num_classes,
            criterion,
            None,
            None,
            aux_weight,
            amp_enabled,
        )
        _sync(device)
        test_seconds = time.perf_counter() - test_started

        prediction_map = np.full(hsi.labels.shape, -1, dtype=np.int16)
        target_map = np.full(hsi.labels.shape, -1, dtype=np.int16)
        rows, cols = test_result["coords"].T
        prediction_map[rows, cols] = test_result["predictions"].astype(np.int16)
        target_map[rows, cols] = test_result["targets"].astype(np.int16)
        np.save(run_dir / "prediction.npy", prediction_map)
        np.save(run_dir / "ground_truth_eval.npy", target_map)
        np.save(run_dir / "confusion_matrix.npy", test_result["confusion"])
        np.save(run_dir / "gate.npy", test_result["gate"].astype(np.float32))
        np.save(run_dir / "spatial_entropy.npy", test_result["spatial_entropy"].astype(np.float32))
        np.save(run_dir / "spectral_entropy.npy", test_result["spectral_entropy"].astype(np.float32))
        save_confusion_csv(run_dir / "confusion_matrix.csv", test_result["confusion"])

        finite_gate = test_result["gate"][np.isfinite(test_result["gate"])]
        gate_mean = float(finite_gate.mean()) if finite_gate.size else None
        gate_std = float(finite_gate.std()) if finite_gate.size else None
        metrics = {
            **test_result["metrics"],
            "test_loss": test_result["loss"],
            "best_val_oa": best_oa,
            "best_epoch": best_epoch,
            "epochs_completed": len(curves),
            "parameter_count": parameter_count,
            "training_seconds": training_seconds,
            "test_inference_seconds": test_seconds,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "seed": seed,
            "dataset": config["data"]["name"],
            "protocol": protocol_name,
            "model": _model_name(config),
        }
        json_dump(run_dir / "metrics.json", metrics)
        json_dump(run_dir / "environment.json", environment_snapshot())
        json_dump(run_dir / "status.json", {"state": "success", "seed": seed})

        output_files = [path for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"]
        manifest = build_manifest(
            run_dir,
            input_files={
                "data": data_cfg["data_path"],
                "ground_truth": data_cfg["gt_path"],
                "split": split_path,
            },
            output_files=output_files,
            extra={
                "status": "success",
                "seed": seed,
                "dataset": config["data"]["name"],
                "protocol": protocol_name,
                "model": _model_name(config),
            },
        )
        json_dump(run_dir / "manifest.json", manifest)
        return run_dir
    except Exception as exc:
        json_dump(
            run_dir / "status.json",
            {
                "state": "failed",
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
