from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class StabilityThresholds:
    positive_repeat_fraction: float = 2.0 / 3.0
    min_mean_advantage: float = 0.05
    min_worst_advantage: float = -0.05
    min_stable_classes_per_seed: int = 3
    min_seed_pass_fraction: float = 0.8
    min_mean_global_oa_gap: float = -0.05

    def __post_init__(self) -> None:
        if not 0 < self.positive_repeat_fraction <= 1:
            raise ValueError("positive_repeat_fraction must be in (0, 1]")
        if self.min_mean_advantage < 0:
            raise ValueError("min_mean_advantage must be non-negative")
        if self.min_worst_advantage > 0:
            raise ValueError("min_worst_advantage must be non-positive")
        if self.min_stable_classes_per_seed < 1:
            raise ValueError("min_stable_classes_per_seed must be positive")
        if not 0 < self.min_seed_pass_fraction <= 1:
            raise ValueError("min_seed_pass_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_repeat_fraction": float(self.positive_repeat_fraction),
            "min_mean_advantage": float(self.min_mean_advantage),
            "min_worst_advantage": float(self.min_worst_advantage),
            "min_stable_classes_per_seed": int(self.min_stable_classes_per_seed),
            "min_seed_pass_fraction": float(self.min_seed_pass_fraction),
            "min_mean_global_oa_gap": float(self.min_mean_global_oa_gap),
        }


def _prediction_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have shape (at least two repeats, samples)")
    return matrix


def repeated_oof_stability_profile(
    targets: np.ndarray,
    spatial_predictions: np.ndarray,
    spectral_predictions: np.ndarray,
    num_classes: int,
    *,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    """Audit repeat-stable spectral advantage without using test labels."""

    thresholds = thresholds or StabilityThresholds()
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    spatial = _prediction_matrix(spatial_predictions, name="spatial_predictions")
    spectral = _prediction_matrix(spectral_predictions, name="spectral_predictions")
    if spatial.shape != spectral.shape or spatial.shape[1] != len(targets):
        raise ValueError("targets and repeated prediction matrices must align")
    if np.any((targets < 0) | (targets >= num_classes)):
        raise ValueError("targets contain an out-of-range class index")

    repeats = int(spatial.shape[0])
    required_positive_repeats = int(ceil(thresholds.positive_repeat_fraction * repeats))
    spatial_correct = spatial == targets[None, :]
    spectral_correct = spectral == targets[None, :]
    spectral_only = spectral_correct & ~spatial_correct
    spatial_only = spatial_correct & ~spectral_correct

    stable_spectral_only = spectral_only.sum(axis=0) >= required_positive_repeats
    stable_spatial_only = spatial_only.sum(axis=0) >= required_positive_repeats
    per_class: list[dict[str, Any]] = []
    stable_classes: list[int] = []
    for class_id in range(num_classes):
        selected = targets == class_id
        count = int(selected.sum())
        if count == 0:
            repeat_net = np.zeros(repeats, dtype=np.float64)
            repeat_advantage = repeat_net.copy()
        else:
            repeat_net = (
                spectral_only[:, selected].sum(axis=1)
                - spatial_only[:, selected].sum(axis=1)
            ).astype(np.float64)
            repeat_advantage = repeat_net / count
        positive_repeats = int(np.sum(repeat_advantage > 0))
        mean_advantage = float(repeat_advantage.mean())
        worst_advantage = float(repeat_advantage.min())
        stable_positive = bool(
            count > 0
            and positive_repeats >= required_positive_repeats
            and mean_advantage >= thresholds.min_mean_advantage
            and worst_advantage >= thresholds.min_worst_advantage
        )
        if stable_positive:
            stable_classes.append(class_id)
        stable_beneficial_count = int(np.sum(stable_spectral_only & selected))
        stable_harmful_count = int(np.sum(stable_spatial_only & selected))
        per_class.append(
            {
                "class_id": class_id,
                "count": count,
                "repeat_net_correct": repeat_net.astype(float).tolist(),
                "repeat_advantage": repeat_advantage.astype(float).tolist(),
                "positive_repeats": positive_repeats,
                "required_positive_repeats": required_positive_repeats,
                "mean_advantage": mean_advantage,
                "worst_advantage": worst_advantage,
                "stable_beneficial_count": stable_beneficial_count,
                "stable_harmful_count": stable_harmful_count,
                "stable_net_correct": stable_beneficial_count - stable_harmful_count,
                "stable_positive": stable_positive,
            }
        )

    spatial_oa_by_repeat = spatial_correct.mean(axis=1)
    spectral_oa_by_repeat = spectral_correct.mean(axis=1)
    stable_beneficial_count = int(stable_spectral_only.sum())
    stable_harmful_count = int(stable_spatial_only.sum())
    stable_class_count = int(len(stable_classes))
    seed_qualified = bool(
        stable_class_count >= thresholds.min_stable_classes_per_seed
        and stable_beneficial_count > stable_harmful_count
    )
    return {
        "repeats": repeats,
        "sample_count": int(len(targets)),
        "required_positive_repeats": required_positive_repeats,
        "spatial_oa_by_repeat": spatial_oa_by_repeat.astype(float).tolist(),
        "spectral_oa_by_repeat": spectral_oa_by_repeat.astype(float).tolist(),
        "spatial_oa_mean": float(spatial_oa_by_repeat.mean()),
        "spectral_oa_mean": float(spectral_oa_by_repeat.mean()),
        "global_oa_gap": float(spectral_oa_by_repeat.mean() - spatial_oa_by_repeat.mean()),
        "stable_class_ids": stable_classes,
        "stable_class_count": stable_class_count,
        "stable_beneficial_count": stable_beneficial_count,
        "stable_harmful_count": stable_harmful_count,
        "stable_net_correct": stable_beneficial_count - stable_harmful_count,
        "seed_qualified": seed_qualified,
        "per_class": per_class,
        "thresholds": thresholds.to_dict(),
    }


def evaluate_encoder_profiles(
    encoder: str,
    seed_profiles: Iterable[dict[str, Any]],
    *,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or StabilityThresholds()
    profiles = list(seed_profiles)
    if not profiles:
        raise ValueError("seed_profiles must not be empty")
    required_seed_passes = int(ceil(thresholds.min_seed_pass_fraction * len(profiles)))
    qualified_seed_count = int(sum(bool(row["seed_qualified"]) for row in profiles))
    stable_class_counts = np.asarray(
        [row["stable_class_count"] for row in profiles], dtype=np.float64
    )
    global_oa_gaps = np.asarray([row["global_oa_gap"] for row in profiles], dtype=np.float64)
    stable_net = np.asarray([row["stable_net_correct"] for row in profiles], dtype=np.float64)
    checks = {
        "enough_qualified_seeds": qualified_seed_count >= required_seed_passes,
        "mean_global_oa_gap": float(global_oa_gaps.mean())
        >= thresholds.min_mean_global_oa_gap,
        "positive_mean_stable_net": float(stable_net.mean()) > 0,
    }
    return {
        "encoder": str(encoder),
        "seed_count": len(profiles),
        "required_seed_passes": required_seed_passes,
        "qualified_seed_count": qualified_seed_count,
        "stable_class_count_mean": float(stable_class_counts.mean()),
        "stable_class_count_median": float(np.median(stable_class_counts)),
        "global_oa_gap_mean": float(global_oa_gaps.mean()),
        "stable_net_correct_mean": float(stable_net.mean()),
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "passed": bool(all(checks.values())),
        "thresholds": thresholds.to_dict(),
    }


def select_encoder(evaluations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(evaluations)
    if not rows:
        raise ValueError("evaluations must not be empty")
    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["passed"]),
            int(row["qualified_seed_count"]),
            float(row["stable_class_count_median"]),
            float(row["stable_net_correct_mean"]),
            float(row["global_oa_gap_mean"]),
            str(row["encoder"]) == "conv1d",
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked[0]["passed"] else None
    return {
        "decision": "DEVELOPMENT_GO" if selected is not None else "DEVELOPMENT_NO_GO",
        "selected_encoder": selected["encoder"] if selected is not None else None,
        "diagnostic_best_encoder": ranked[0]["encoder"],
        "evaluations": rows,
    }
