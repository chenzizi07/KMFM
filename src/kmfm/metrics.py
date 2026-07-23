from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix_from_arrays(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    valid = (
        (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    encoded = targets[valid] * num_classes + predictions[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("confusion matrix contains no samples")
    diagonal = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    per_class = np.divide(
        diagonal,
        support,
        out=np.full_like(diagonal, np.nan, dtype=np.float64),
        where=support > 0,
    )
    oa = float(diagonal.sum() / total)
    aa = float(np.nanmean(per_class))
    expected = float(np.dot(support, predicted) / (total * total))
    kappa = float((oa - expected) / (1.0 - expected)) if expected < 1.0 else float("nan")
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_accuracy": per_class.tolist(),
        "support": support.astype(np.int64).tolist(),
        "num_samples": total,
    }


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> tuple[dict[str, Any], np.ndarray]:
    confusion = confusion_matrix_from_arrays(targets, predictions, num_classes)
    return metrics_from_confusion(confusion), confusion


def probabilistic_metrics(
    targets: np.ndarray, logits: np.ndarray, num_bins: int = 15
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits must have shape (samples, classes) and match targets")
    if logits.shape[1] < 2:
        raise ValueError("probabilistic metrics require at least two classes")
    if np.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("targets contain an invalid class index")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")

    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    selected = probabilities[np.arange(len(targets)), targets].clip(1e-12, 1.0)
    nll = float(-np.log(selected).mean())
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[targets]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())

    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == targets
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for index in range(num_bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if index == 0:
            in_bin |= confidence == 0.0
        if not np.any(in_bin):
            continue
        bin_accuracy = float(correctness[in_bin].mean())
        bin_confidence = float(confidence[in_bin].mean())
        ece += float(in_bin.mean()) * abs(bin_accuracy - bin_confidence)
    return {"nll": nll, "brier": brier, "ece": float(ece)}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def routing_diagnostics(
    targets: np.ndarray,
    fused_predictions: np.ndarray,
    spatial_predictions: np.ndarray,
    spectral_predictions: np.ndarray,
    spatial_weights: np.ndarray,
) -> dict[str, float | int | None]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    fused = np.asarray(fused_predictions, dtype=np.int64).reshape(-1)
    spatial = np.asarray(spatial_predictions, dtype=np.int64).reshape(-1)
    spectral = np.asarray(spectral_predictions, dtype=np.int64).reshape(-1)
    weights = np.asarray(spatial_weights, dtype=np.float64).reshape(-1)
    if not (targets.shape == fused.shape == spatial.shape == spectral.shape == weights.shape):
        raise ValueError("routing arrays must have the same one-dimensional shape")

    spatial_correct = spatial == targets
    spectral_correct = spectral == targets
    fused_correct = fused == targets
    discordant = spatial_correct ^ spectral_correct
    finite_discordant = discordant & np.isfinite(weights)
    routing_auc: float | None = None
    if np.any(finite_discordant):
        labels = spatial_correct[finite_discordant].astype(np.int64)
        scores = weights[finite_discordant]
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives > 0 and negatives > 0:
            ranks = _average_ranks(scores)
            routing_auc = float(
                (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                / (positives * negatives)
            )

    spatial_oa = float(spatial_correct.mean())
    spectral_oa = float(spectral_correct.mean())
    fused_oa = float(fused_correct.mean())
    oracle_oa = float((spatial_correct | spectral_correct).mean())
    best_branch_oa = max(spatial_oa, spectral_oa)
    oracle_gap = oracle_oa - best_branch_oa
    recovery = (fused_oa - best_branch_oa) / oracle_gap if oracle_gap > 1e-12 else None
    return {
        "spatial_branch_oa": spatial_oa,
        "spectral_branch_oa": spectral_oa,
        "oracle_oa": oracle_oa,
        "oracle_gap": float(oracle_gap),
        "oracle_gap_recovery": float(recovery) if recovery is not None else None,
        "discordant_count": int(discordant.sum()),
        "discordant_fraction": float(discordant.mean()),
        "routing_auc": routing_auc,
    }
