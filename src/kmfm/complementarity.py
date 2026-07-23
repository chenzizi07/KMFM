from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .metrics import classification_metrics


def _labels(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative class indices")
    return array


def _predictions(values: np.ndarray, name: str, length: int) -> np.ndarray:
    array = _labels(values, name)
    if len(array) != length:
        raise ValueError(
            f"{name} must contain the same number of samples as targets ({length})"
        )
    return array


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def _metric_row(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    metrics, _ = classification_metrics(targets, predictions, num_classes)
    return {
        "oa": float(metrics["oa"]),
        "aa": float(metrics["aa"]),
        "kappa": float(metrics["kappa"]),
        "per_class_accuracy": [float(value) for value in metrics["per_class_accuracy"]],
    }


def analyze_seed(
    targets: np.ndarray,
    spatial_predictions: np.ndarray,
    spectral_predictions: np.ndarray,
    *,
    global_predictions: np.ndarray | None = None,
    adlf_predictions: np.ndarray | None = None,
    selected_alpha: float | None = None,
    selected_radius: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = _labels(targets, "targets")
    spatial = _predictions(spatial_predictions, "spatial_predictions", len(labels))
    spectral = _predictions(spectral_predictions, "spectral_predictions", len(labels))
    optional = {
        "global": global_predictions,
        "adlf": adlf_predictions,
    }
    validated_optional: dict[str, np.ndarray] = {}
    for name, values in optional.items():
        if values is not None:
            validated_optional[name] = _predictions(values, f"{name}_predictions", len(labels))

    num_classes = int(
        max(
            labels.max(initial=0),
            spatial.max(initial=0),
            spectral.max(initial=0),
            *(values.max(initial=0) for values in validated_optional.values()),
        )
        + 1
    )
    spatial_correct = spatial == labels
    spectral_correct = spectral == labels
    oracle_correct = spatial_correct | spectral_correct
    exclusive_spatial = spatial_correct & ~spectral_correct
    exclusive_spectral = spectral_correct & ~spatial_correct
    both_correct = spatial_correct & spectral_correct
    both_wrong = ~spatial_correct & ~spectral_correct
    spatial_metrics = _metric_row(labels, spatial, num_classes)
    spectral_metrics = _metric_row(labels, spectral, num_classes)
    oracle_predictions = np.where(spatial_correct, spatial, spectral)
    oracle_metrics = _metric_row(labels, oracle_predictions, num_classes)

    class_support = np.bincount(labels, minlength=num_classes).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for class_index in range(num_classes):
        mask = labels == class_index
        support = int(mask.sum())
        row: dict[str, Any] = {
            "class_index": class_index,
            "support": support,
            "support_fraction": float(support / len(labels)),
            "spatial_correct": int((spatial_correct & mask).sum()),
            "spectral_correct": int((spectral_correct & mask).sum()),
            "oracle_correct": int((oracle_correct & mask).sum()),
            "exclusive_spatial_correct": int((exclusive_spatial & mask).sum()),
            "exclusive_spectral_correct": int((exclusive_spectral & mask).sum()),
            "both_correct": int((both_correct & mask).sum()),
            "both_wrong": int((both_wrong & mask).sum()),
        }
        for name, predictions in validated_optional.items():
            correct = predictions == labels
            row[f"{name}_correct"] = int((correct & mask).sum())
            row[f"{name}_improved_vs_spatial"] = int((~spatial_correct & correct & mask).sum())
            row[f"{name}_harmed_vs_spatial"] = int((spatial_correct & ~correct & mask).sum())
        for name in ("spatial", "spectral", "oracle", *validated_optional):
            correct_count = row[f"{name}_correct"]
            row[f"{name}_accuracy"] = float(correct_count / support) if support else None
        row["oracle_gain_vs_spatial_pp"] = float(
            100.0 * (row["oracle_accuracy"] - row["spatial_accuracy"])
        )
        rows.append(row)

    global_metrics = (
        _metric_row(labels, validated_optional["global"], num_classes)
        if "global" in validated_optional
        else None
    )
    adlf_metrics = (
        _metric_row(labels, validated_optional["adlf"], num_classes)
        if "adlf" in validated_optional
        else None
    )
    global_delta = (
        np.asarray(global_metrics["per_class_accuracy"], dtype=float)
        - np.asarray(spatial_metrics["per_class_accuracy"], dtype=float)
        if global_metrics is not None
        else None
    )
    seed_row: dict[str, Any] = {
        "num_samples": len(labels),
        "num_classes": num_classes,
        "spatial_oa": spatial_metrics["oa"],
        "spatial_aa": spatial_metrics["aa"],
        "spatial_kappa": spatial_metrics["kappa"],
        "spectral_oa": spectral_metrics["oa"],
        "spectral_aa": spectral_metrics["aa"],
        "spectral_kappa": spectral_metrics["kappa"],
        "oracle_oa": oracle_metrics["oa"],
        "oracle_aa": oracle_metrics["aa"],
        "oracle_kappa": oracle_metrics["kappa"],
        "oracle_gain_vs_spatial_pp": 100.0 * (oracle_metrics["oa"] - spatial_metrics["oa"]),
        "oracle_aa_gain_vs_spatial_pp": 100.0 * (oracle_metrics["aa"] - spatial_metrics["aa"]),
        "exclusive_spatial_correct": int(exclusive_spatial.sum()),
        "exclusive_spectral_correct": int(exclusive_spectral.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "discordant_predictions": int((spatial != spectral).sum()),
        "discordant_prediction_fraction": float((spatial != spectral).mean()),
        "correctness_disagreement": int((exclusive_spatial | exclusive_spectral).sum()),
        "correctness_disagreement_fraction": float(
            (exclusive_spatial | exclusive_spectral).mean()
        ),
        "selected_alpha": selected_alpha,
        "selected_radius": selected_radius,
    }
    for name, metrics in (("global", global_metrics), ("adlf", adlf_metrics)):
        if metrics is None:
            continue
        predictions = validated_optional[name]
        correct = predictions == labels
        seed_row.update(
            {
                f"{name}_oa": metrics["oa"],
                f"{name}_aa": metrics["aa"],
                f"{name}_kappa": metrics["kappa"],
                f"{name}_oa_gain_vs_spatial_pp": 100.0 * (metrics["oa"] - spatial_metrics["oa"]),
                f"{name}_aa_gain_vs_spatial_pp": 100.0 * (metrics["aa"] - spatial_metrics["aa"]),
                f"{name}_oracle_recovery": (
                    (metrics["oa"] - spatial_metrics["oa"])
                    / (oracle_metrics["oa"] - spatial_metrics["oa"])
                    if oracle_metrics["oa"] > spatial_metrics["oa"]
                    else None
                ),
                f"{name}_improved_vs_spatial": int((~spatial_correct & correct).sum()),
                f"{name}_harmed_vs_spatial": int((spatial_correct & ~correct).sum()),
                f"{name}_net_corrected_vs_spatial": int(
                    ((~spatial_correct & correct).sum() - (spatial_correct & ~correct).sum())
                ),
            }
        )
    seed_row["global_aa_positive_oa_negative"] = bool(
        seed_row.get("global_aa_gain_vs_spatial_pp", 0.0) > 0.0
        and seed_row.get("global_oa_gain_vs_spatial_pp", 0.0) < 0.0
    )
    seed_row["global_support_gain_spearman"] = (
        _safe_spearman(class_support.astype(float), global_delta)
        if global_delta is not None
        else None
    )
    return seed_row, rows


def summarize_class_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame_columns = sorted({key for row in rows for key in row})
    numeric_columns = {
        key
        for key in frame_columns
        if key not in {"class_index", "seed"} and all(
            row.get(key) is None or isinstance(row.get(key), (int, float, np.integer, np.floating))
            for row in rows
        )
    }
    summary: list[dict[str, Any]] = []
    for class_index in sorted({int(row["class_index"]) for row in rows}):
        group = [row for row in rows if int(row["class_index"]) == class_index]
        record: dict[str, Any] = {"class_index": class_index, "n_seeds": len(group)}
        for key in sorted(numeric_columns - {"class_index"}):
            values = np.asarray(
                [float(row[key]) for row in group if row.get(key) is not None], dtype=float
            )
            if not len(values):
                continue
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(record)
    return summary
