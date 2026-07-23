from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class OOFAdvantageProfile:
    class_weights: np.ndarray
    per_class: tuple[dict[str, Any], ...]
    spatial_oa: float
    spectral_oa: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_weights": self.class_weights.astype(float).tolist(),
            "per_class": list(self.per_class),
            "spatial_oa": float(self.spatial_oa),
            "spectral_oa": float(self.spectral_oa),
        }


def stratified_folds(
    targets: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Return deterministic, exhaustive class-stratified holdout indices."""

    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if targets.size == 0:
        raise ValueError("targets must not be empty")
    rng = np.random.default_rng(seed)
    fold_members: list[list[int]] = [[] for _ in range(n_splits)]
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        if len(indices) < n_splits:
            raise ValueError(
                f"Class {int(class_id)} has {len(indices)} samples, fewer than {n_splits} folds"
            )
        shuffled = rng.permutation(indices)
        for fold_id, chunk in enumerate(np.array_split(shuffled, n_splits)):
            fold_members[fold_id].extend(int(index) for index in chunk)
    folds = tuple(np.asarray(sorted(members), dtype=np.int64) for members in fold_members)
    combined = np.concatenate(folds)
    if len(combined) != len(targets) or not np.array_equal(
        np.sort(combined), np.arange(len(targets), dtype=np.int64)
    ):
        raise RuntimeError("Stratified folds are not an exhaustive partition")
    return folds


def class_advantage_profile(
    targets: np.ndarray,
    spatial_predictions: np.ndarray,
    spectral_predictions: np.ndarray,
    num_classes: int,
    *,
    prior_strength: float = 4.0,
    reference_gain: float = 0.25,
) -> OOFAdvantageProfile:
    """Estimate shrunk class-level spectral advantage from OOF predictions.

    The raw paired advantage for class c is (spectral-only correct minus
    spatial-only correct) / n_c. Four zero-advantage pseudo-observations shrink
    the estimate, and a fixed 25 percentage-point gain maps to unit weight.
    """

    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    if reference_gain <= 0:
        raise ValueError("reference_gain must be positive")
    arrays = [
        np.asarray(value, dtype=np.int64).reshape(-1)
        for value in (targets, spatial_predictions, spectral_predictions)
    ]
    if len({len(value) for value in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError("targets and predictions must have the same non-zero length")
    targets, spatial_predictions, spectral_predictions = arrays
    if np.any((targets < 0) | (targets >= num_classes)):
        raise ValueError("targets contain an out-of-range class index")

    rows: list[dict[str, Any]] = []
    weights = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        selected = targets == class_id
        count = int(selected.sum())
        spatial_correct = spatial_predictions[selected] == class_id
        spectral_correct = spectral_predictions[selected] == class_id
        spectral_only = int(np.sum(spectral_correct & ~spatial_correct))
        spatial_only = int(np.sum(spatial_correct & ~spectral_correct))
        net = spectral_only - spatial_only
        raw_advantage = float(net / count) if count else 0.0
        shrunk_advantage = float(net / (count + prior_strength)) if count else 0.0
        weight = float(np.clip(max(0.0, shrunk_advantage) / reference_gain, 0.0, 1.0))
        weights[class_id] = weight
        rows.append(
            {
                "class_id": class_id,
                "count": count,
                "spatial_correct": int(spatial_correct.sum()),
                "spectral_correct": int(spectral_correct.sum()),
                "spectral_only_correct": spectral_only,
                "spatial_only_correct": spatial_only,
                "raw_advantage": raw_advantage,
                "shrunk_advantage": shrunk_advantage,
                "distillation_weight": weight,
            }
        )
    return OOFAdvantageProfile(
        class_weights=weights,
        per_class=tuple(rows),
        spatial_oa=float(np.mean(spatial_predictions == targets)),
        spectral_oa=float(np.mean(spectral_predictions == targets)),
    )


def advantage_weighted_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL distillation with fixed class weights and a detached teacher."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    if student_logits.ndim != 2 or targets.shape != student_logits.shape[:1]:
        raise ValueError("Expected logits (batch, classes) and targets (batch,)")
    if class_weights.ndim != 1 or class_weights.numel() != student_logits.shape[1]:
        raise ValueError("class_weights must contain one value per class")
    sample_weights = class_weights.to(student_logits.device, student_logits.dtype)[targets]
    softened_teacher = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    per_sample = F.kl_div(
        torch.log_softmax(student_logits / temperature, dim=-1),
        softened_teacher,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)
    denominator = sample_weights.sum().clamp_min(1.0)
    return (per_sample * sample_weights).sum() / denominator, sample_weights.mean()
