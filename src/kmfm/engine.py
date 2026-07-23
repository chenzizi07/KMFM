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
from scipy import stats
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.utils.data import DataLoader, Subset

from .artifacts import build_manifest, environment_snapshot, json_dump, save_confusion_csv
from .data import HSIPatchDataset, load_hsi, standardize_cube
from .distillation import (
    advantage_weighted_distillation_loss,
    class_advantage_profile,
    stratified_folds,
)
from .metrics import classification_metrics, probabilistic_metrics, routing_diagnostics
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


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def _finite_summary(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    return float(finite.mean()), float(finite.std())


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return None
    result = stats.spearmanr(x[finite], y[finite])
    correlation = float(result.statistic)
    return correlation if np.isfinite(correlation) else None


def _model_name(config: dict[str, Any]) -> str:
    model = config["model"]
    return model.get(
        "name",
        f"lassf_{model.get('spectral', 'conv1d')}_{model.get('fusion', 'reliability')}_h{model.get('hidden_dim', 64)}",
    )


def _build_model(
    *,
    bands: int,
    num_classes: int,
    model_cfg: dict[str, Any],
    device: torch.device,
) -> LASSFNet:
    return LASSFNet(
        bands=bands,
        num_classes=num_classes,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        spectral=model_cfg.get("spectral", "conv1d"),
        fusion=model_cfg.get("fusion", "reliability"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        normalize_branches=bool(model_cfg.get("normalize_branches", False)),
        entropy_temperature=float(model_cfg.get("entropy_temperature", 0.25)),
    ).to(device)


def _make_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler | None:
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


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
    distillation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    distillation_losses: list[float] = []
    distillation_weights: list[float] = []
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    coords_all: list[np.ndarray] = []
    gates_all: list[np.ndarray] = []
    spatial_entropy_all: list[np.ndarray] = []
    spectral_entropy_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    spatial_logits_all: list[np.ndarray] = []
    spectral_logits_all: list[np.ndarray] = []
    spatial_calibrated_logits_all: list[np.ndarray] = []
    spectral_calibrated_logits_all: list[np.ndarray] = []
    spatial_feature_norm_all: list[np.ndarray] = []
    spectral_feature_norm_all: list[np.ndarray] = []
    spatial_contribution_norm_all: list[np.ndarray] = []
    spectral_contribution_norm_all: list[np.ndarray] = []
    contribution_ratio_all: list[np.ndarray] = []

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
                distillation_loss = torch.zeros((), device=device, dtype=loss.dtype)
                mean_distillation_weight = torch.zeros((), device=device, dtype=loss.dtype)
                if training and distillation is not None:
                    distillation_loss, mean_distillation_weight = (
                        advantage_weighted_distillation_loss(
                            output["logits"],
                            output["spectral_logits"],
                            target,
                            distillation["class_weights"],
                            temperature=float(distillation["temperature"]),
                        )
                    )
                    loss = loss + float(distillation["coefficient"]) * distillation_loss
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            distillation_losses.append(float(distillation_loss.detach().cpu()))
            distillation_weights.append(float(mean_distillation_weight.detach().cpu()))
            targets_all.append(target.detach().cpu().numpy())
            predictions_all.append(output["logits"].argmax(dim=-1).detach().cpu().numpy())
            coords_all.append(coords.numpy())
            gates_all.append(output["gate"].detach().float().cpu().numpy())
            spatial_entropy_all.append(output["spatial_entropy"].detach().float().cpu().numpy())
            spectral_entropy_all.append(output["spectral_entropy"].detach().float().cpu().numpy())
            logits_all.append(output["logits"].detach().float().cpu().numpy())
            spatial_logits_all.append(output["spatial_logits"].detach().float().cpu().numpy())
            spectral_logits_all.append(output["spectral_logits"].detach().float().cpu().numpy())
            spatial_calibrated_logits_all.append(
                output["spatial_calibrated_logits"].detach().float().cpu().numpy()
            )
            spectral_calibrated_logits_all.append(
                output["spectral_calibrated_logits"].detach().float().cpu().numpy()
            )
            spatial_feature_norm_all.append(
                output["spatial_feature_norm"].detach().float().cpu().numpy()
            )
            spectral_feature_norm_all.append(
                output["spectral_feature_norm"].detach().float().cpu().numpy()
            )
            spatial_contribution_norm_all.append(
                output["spatial_contribution_norm"].detach().float().cpu().numpy()
            )
            spectral_contribution_norm_all.append(
                output["spectral_contribution_norm"].detach().float().cpu().numpy()
            )
            contribution_ratio_all.append(
                output["contribution_ratio"].detach().float().cpu().numpy()
            )

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    logits = np.concatenate(logits_all)
    spatial_logits = np.concatenate(spatial_logits_all)
    spectral_logits = np.concatenate(spectral_logits_all)
    spatial_calibrated_logits = np.concatenate(spatial_calibrated_logits_all)
    spectral_calibrated_logits = np.concatenate(spectral_calibrated_logits_all)
    gate = np.concatenate(gates_all)
    spatial_predictions = spatial_logits.argmax(axis=-1)
    spectral_predictions = spectral_logits.argmax(axis=-1)
    metrics, confusion = classification_metrics(targets, predictions, num_classes)
    metrics.update(probabilistic_metrics(targets, logits))
    metrics.update(
        routing_diagnostics(
            targets,
            predictions,
            spatial_predictions,
            spectral_predictions,
            gate,
        )
    )
    spatial_probability_metrics = probabilistic_metrics(targets, spatial_calibrated_logits)
    spectral_probability_metrics = probabilistic_metrics(targets, spectral_calibrated_logits)
    metrics.update(
        {f"spatial_branch_{name}": value for name, value in spatial_probability_metrics.items()}
    )
    metrics.update(
        {f"spectral_branch_{name}": value for name, value in spectral_probability_metrics.items()}
    )
    return {
        "loss": float(np.mean(losses)),
        "distillation_loss": float(np.mean(distillation_losses)),
        "distillation_weight": float(np.mean(distillation_weights)),
        "metrics": metrics,
        "confusion": confusion,
        "targets": targets,
        "predictions": predictions,
        "logits": logits,
        "spatial_predictions": spatial_predictions,
        "spectral_predictions": spectral_predictions,
        "spatial_logits": spatial_logits,
        "spectral_logits": spectral_logits,
        "spatial_calibrated_logits": spatial_calibrated_logits,
        "spectral_calibrated_logits": spectral_calibrated_logits,
        "coords": np.concatenate(coords_all),
        "gate": gate,
        "spatial_entropy": np.concatenate(spatial_entropy_all),
        "spectral_entropy": np.concatenate(spectral_entropy_all),
        "spatial_feature_norm": np.concatenate(spatial_feature_norm_all),
        "spectral_feature_norm": np.concatenate(spectral_feature_norm_all),
        "spatial_contribution_norm": np.concatenate(spatial_contribution_norm_all),
        "spectral_contribution_norm": np.concatenate(spectral_contribution_norm_all),
        "contribution_ratio": np.concatenate(contribution_ratio_all),
    }


def _subset_loader(
    dataset: HSIPatchDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        Subset(dataset, indices.astype(np.int64).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _estimate_oof_advantage(
    *,
    train_dataset: HSIPatchDataset,
    bands: int,
    num_classes: int,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    seed: int,
    distillation_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float]:
    targets = np.asarray(
        [train_dataset.labels[row, col] - 1 for row, col in train_dataset.coords],
        dtype=np.int64,
    )
    n_splits = int(distillation_cfg.get("folds", 3))
    folds = stratified_folds(targets, n_splits=n_splits, seed=seed + 1701)
    spatial_predictions = np.full(len(targets), -1, dtype=np.int64)
    spectral_predictions = np.full(len(targets), -1, dtype=np.int64)
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(distillation_cfg.get("oof_epochs", 60))
    if epochs < 1:
        raise ValueError("distillation.oof_epochs must be positive")
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    aux_weight = float(distillation_cfg.get("oof_aux_weight", 0.5))
    fold_rows: list[dict[str, Any]] = []

    _sync(device)
    started = time.perf_counter()
    all_indices = np.arange(len(targets), dtype=np.int64)
    for fold_id, holdout_indices in enumerate(folds):
        fold_seed = seed * 1009 + fold_id + 1
        set_reproducibility(fold_seed, bool(training_cfg.get("deterministic", True)))
        train_indices = np.setdiff1d(all_indices, holdout_indices, assume_unique=True)
        fold_train_loader = _subset_loader(
            train_dataset,
            train_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=fold_seed,
        )
        fold_holdout_loader = _subset_loader(
            train_dataset,
            holdout_indices,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=fold_seed,
        )
        fold_model_cfg = copy.deepcopy(model_cfg)
        fold_model_cfg["fusion"] = "spatial_only"
        fold_model = _build_model(
            bands=bands,
            num_classes=num_classes,
            model_cfg=fold_model_cfg,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            fold_model.parameters(),
            lr=float(training_cfg.get("lr", 3e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
        scaler = _make_scaler(device, amp_enabled)
        for _ in range(epochs):
            _run_loader(
                fold_model,
                fold_train_loader,
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
            fold_model,
            fold_holdout_loader,
            device,
            num_classes,
            criterion,
            None,
            None,
            aux_weight,
            amp_enabled,
        )
        spatial_predictions[holdout_indices] = holdout["predictions"]
        spectral_predictions[holdout_indices] = holdout["spectral_predictions"]
        fold_rows.append(
            {
                "fold": fold_id,
                "train_count": int(len(train_indices)),
                "holdout_count": int(len(holdout_indices)),
                "spatial_oa": float(np.mean(holdout["predictions"] == holdout["targets"])),
                "spectral_oa": float(
                    np.mean(holdout["spectral_predictions"] == holdout["targets"])
                ),
            }
        )
        del fold_model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _sync(device)
    elapsed = time.perf_counter() - started
    if np.any(spatial_predictions < 0) or np.any(spectral_predictions < 0):
        raise RuntimeError("OOF prediction arrays were not filled completely")
    profile = class_advantage_profile(
        targets,
        spatial_predictions,
        spectral_predictions,
        num_classes,
        prior_strength=float(distillation_cfg.get("prior_strength", 4.0)),
        reference_gain=float(distillation_cfg.get("reference_gain", 0.25)),
    )
    payload = {
        "mode": "oof_class",
        "folds": n_splits,
        "oof_epochs": epochs,
        "oof_aux_weight": aux_weight,
        "prior_strength": float(distillation_cfg.get("prior_strength", 4.0)),
        "reference_gain": float(distillation_cfg.get("reference_gain", 0.25)),
        "fold_results": fold_rows,
        **profile.to_dict(),
    }
    return profile.class_weights, payload, elapsed


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
        criterion = nn.CrossEntropyLoss()
        training_cfg = config["training"]
        distillation_cfg = copy.deepcopy(model_cfg.get("distillation", {"mode": "none"}))
        distillation_mode = str(distillation_cfg.get("mode", "none"))
        if distillation_mode not in {"none", "uniform", "oof_class"}:
            raise ValueError(
                "model.distillation.mode must be one of none, uniform or oof_class"
            )
        oof_training_seconds = 0.0
        if distillation_mode == "oof_class":
            if not isinstance(train_loader.dataset, HSIPatchDataset):
                raise TypeError("OOF distillation requires an HSIPatchDataset training set")
            class_weights, distillation_profile, oof_training_seconds = (
                _estimate_oof_advantage(
                    train_dataset=train_loader.dataset,
                    bands=cube.shape[-1],
                    num_classes=hsi.num_classes,
                    model_cfg=model_cfg,
                    training_cfg=training_cfg,
                    criterion=criterion,
                    device=device,
                    seed=seed,
                    distillation_cfg=distillation_cfg,
                )
            )
        elif distillation_mode == "uniform":
            class_weights = np.ones(hsi.num_classes, dtype=np.float32)
            distillation_profile = {
                "mode": "uniform",
                "class_weights": class_weights.astype(float).tolist(),
                "per_class": [],
                "spatial_oa": None,
                "spectral_oa": None,
            }
        else:
            class_weights = np.zeros(hsi.num_classes, dtype=np.float32)
            distillation_profile = {
                "mode": "none",
                "class_weights": class_weights.astype(float).tolist(),
                "per_class": [],
                "spatial_oa": None,
                "spectral_oa": None,
            }
        coefficient = float(distillation_cfg.get("coefficient", 0.5))
        temperature = float(distillation_cfg.get("temperature", 2.0))
        if coefficient < 0:
            raise ValueError("model.distillation.coefficient must be non-negative")
        if temperature <= 0:
            raise ValueError("model.distillation.temperature must be positive")
        distillation_profile.update(
            {
                "coefficient": coefficient,
                "temperature": temperature,
                "active_class_count": int(np.sum(class_weights > 0.0)),
                "oof_training_seconds": oof_training_seconds,
            }
        )
        json_dump(run_dir / "distillation_profile.json", distillation_profile)

        # Cross-fitting consumes random state. Reset before final training so all
        # v6 variants start from the same seed-specific initialization.
        set_reproducibility(seed, bool(training_cfg.get("deterministic", True)))
        model = _build_model(
            bands=cube.shape[-1],
            num_classes=hsi.num_classes,
            model_cfg=model_cfg,
            device=device,
        )
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("lr", 3e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(training_cfg.get("epochs", 200))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
        scaler = _make_scaler(device, amp_enabled)
        aux_weight = float(training_cfg.get("aux_weight", 0.2))
        patience = int(training_cfg.get("patience", 30))
        distillation_context = (
            {
                "class_weights": torch.as_tensor(
                    class_weights, dtype=torch.float32, device=device
                ),
                "coefficient": coefficient,
                "temperature": temperature,
            }
            if distillation_mode != "none" and coefficient > 0
            else None
        )

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
                distillation_context,
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
                    "train_distillation_loss": train_result["distillation_loss"],
                    "train_distillation_weight": train_result["distillation_weight"],
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

        calibration = {
            "enabled": False,
            "spatial_temperature": 1.0,
            "spectral_temperature": 1.0,
        }
        if bool(model_cfg.get("calibrate_branch_temperatures", False)):
            validation_before = _run_loader(
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
            spatial_temperature = _fit_temperature(
                validation_before["spatial_logits"], validation_before["targets"]
            )
            spectral_temperature = _fit_temperature(
                validation_before["spectral_logits"], validation_before["targets"]
            )
            model.set_branch_temperatures(spatial_temperature, spectral_temperature)
            validation_after = _run_loader(
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
            calibration = {
                "enabled": True,
                "spatial_temperature": spatial_temperature,
                "spectral_temperature": spectral_temperature,
                "spatial_nll_before": validation_before["metrics"]["spatial_branch_nll"],
                "spatial_nll_after": validation_after["metrics"]["spatial_branch_nll"],
                "spectral_nll_before": validation_before["metrics"]["spectral_branch_nll"],
                "spectral_nll_after": validation_after["metrics"]["spectral_branch_nll"],
                "fused_oa_before": validation_before["metrics"]["oa"],
                "fused_oa_after": validation_after["metrics"]["oa"],
                "fused_nll_before": validation_before["metrics"]["nll"],
                "fused_nll_after": validation_after["metrics"]["nll"],
            }
        json_dump(run_dir / "calibration.json", calibration)

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
        np.save(run_dir / "test_coords.npy", test_result["coords"].astype(np.int32))
        np.save(run_dir / "test_targets.npy", test_result["targets"].astype(np.int16))
        np.save(run_dir / "test_predictions.npy", test_result["predictions"].astype(np.int16))
        np.save(run_dir / "test_logits.npy", test_result["logits"].astype(np.float32))
        np.save(
            run_dir / "spatial_predictions.npy",
            test_result["spatial_predictions"].astype(np.int16),
        )
        np.save(
            run_dir / "spectral_predictions.npy",
            test_result["spectral_predictions"].astype(np.int16),
        )
        np.save(run_dir / "spatial_logits.npy", test_result["spatial_logits"].astype(np.float32))
        np.save(run_dir / "spectral_logits.npy", test_result["spectral_logits"].astype(np.float32))
        np.save(
            run_dir / "spatial_calibrated_logits.npy",
            test_result["spatial_calibrated_logits"].astype(np.float32),
        )
        np.save(
            run_dir / "spectral_calibrated_logits.npy",
            test_result["spectral_calibrated_logits"].astype(np.float32),
        )
        np.save(run_dir / "gate.npy", test_result["gate"].astype(np.float32))
        np.save(run_dir / "spatial_entropy.npy", test_result["spatial_entropy"].astype(np.float32))
        np.save(run_dir / "spectral_entropy.npy", test_result["spectral_entropy"].astype(np.float32))
        np.save(
            run_dir / "spatial_feature_norm.npy",
            test_result["spatial_feature_norm"].astype(np.float32),
        )
        np.save(
            run_dir / "spectral_feature_norm.npy",
            test_result["spectral_feature_norm"].astype(np.float32),
        )
        np.save(
            run_dir / "spatial_contribution_norm.npy",
            test_result["spatial_contribution_norm"].astype(np.float32),
        )
        np.save(
            run_dir / "spectral_contribution_norm.npy",
            test_result["spectral_contribution_norm"].astype(np.float32),
        )
        np.save(
            run_dir / "contribution_ratio.npy",
            test_result["contribution_ratio"].astype(np.float32),
        )
        if protocol_name == "spatial_block":
            boundary_distance_map = distance_transform_edt(split.region_map == TEST)
            boundary_distance = boundary_distance_map[rows, cols]
        else:
            boundary_distance = np.full(len(rows), np.nan, dtype=np.float32)
        np.save(run_dir / "boundary_distance.npy", boundary_distance.astype(np.float32))
        save_confusion_csv(run_dir / "confusion_matrix.csv", test_result["confusion"])

        gate_mean, gate_std = _finite_summary(test_result["gate"])
        spatial_feature_norm_mean, spatial_feature_norm_std = _finite_summary(
            test_result["spatial_feature_norm"]
        )
        spectral_feature_norm_mean, spectral_feature_norm_std = _finite_summary(
            test_result["spectral_feature_norm"]
        )
        spatial_contribution_norm_mean, spatial_contribution_norm_std = _finite_summary(
            test_result["spatial_contribution_norm"]
        )
        spectral_contribution_norm_mean, spectral_contribution_norm_std = _finite_summary(
            test_result["spectral_contribution_norm"]
        )
        contribution_ratio_mean, contribution_ratio_std = _finite_summary(
            test_result["contribution_ratio"]
        )
        entropy_gap = test_result["spectral_entropy"] - test_result["spatial_entropy"]
        metrics = {
            **test_result["metrics"],
            "test_loss": test_result["loss"],
            "best_val_oa": best_oa,
            "best_epoch": best_epoch,
            "epochs_completed": len(curves),
            "parameter_count": parameter_count,
            "training_seconds": training_seconds,
            "oof_training_seconds": oof_training_seconds,
            "total_training_seconds": training_seconds + oof_training_seconds,
            "test_inference_seconds": test_seconds,
            "distillation_mode": distillation_mode,
            "distillation_coefficient": coefficient,
            "distillation_temperature": temperature,
            "distillation_active_class_count": int(np.sum(class_weights > 0.0)),
            "distillation_class_weight_mean": float(class_weights.mean()),
            "distillation_class_weight_max": float(class_weights.max()),
            "distillation_oof_spatial_oa": distillation_profile.get("spatial_oa"),
            "distillation_oof_spectral_oa": distillation_profile.get("spectral_oa"),
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "spatial_feature_norm_mean": spatial_feature_norm_mean,
            "spatial_feature_norm_std": spatial_feature_norm_std,
            "spectral_feature_norm_mean": spectral_feature_norm_mean,
            "spectral_feature_norm_std": spectral_feature_norm_std,
            "spatial_contribution_norm_mean": spatial_contribution_norm_mean,
            "spatial_contribution_norm_std": spatial_contribution_norm_std,
            "spectral_contribution_norm_mean": spectral_contribution_norm_mean,
            "spectral_contribution_norm_std": spectral_contribution_norm_std,
            "contribution_ratio_mean": contribution_ratio_mean,
            "contribution_ratio_std": contribution_ratio_std,
            "gate_entropy_gap_spearman": _spearman(test_result["gate"], entropy_gap),
            "spatial_temperature": calibration["spatial_temperature"],
            "spectral_temperature": calibration["spectral_temperature"],
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
