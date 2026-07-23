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
