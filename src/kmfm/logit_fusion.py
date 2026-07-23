from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar


DEFAULT_ALPHA_GRID = tuple(np.linspace(0.0, 1.0, 21).tolist())
DEFAULT_RADIUS_GRID = (0.0, 0.05, 0.10, 0.15)


@dataclass(frozen=True)
class ADLFPolicy:
    spatial_temperature: float
    spectral_temperature: float
    global_alpha: float
    residual_radius: float
    margin_scale: float
    validation_nll: dict[str, float]
    alpha_grid: tuple[float, ...]
    radius_grid: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_logits(logits: np.ndarray, targets: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape (samples, classes) with at least two classes")
    if not np.isfinite(values).all():
        raise ValueError("logits contain NaN or infinite values")
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        if len(labels) != len(values):
            raise ValueError("targets and logits must contain the same number of samples")
        if np.any((labels < 0) | (labels >= values.shape[1])):
            raise ValueError("targets contain an invalid class index")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    values = _validate_logits(logits)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray) -> float:
    values = _validate_logits(logits, targets)
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = softmax(values)
    selected = probabilities[np.arange(len(labels)), labels].clip(1e-12, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    values = _validate_logits(logits, targets)
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)

    result = minimize_scalar(
        lambda log_temperature: nll(values / np.exp(log_temperature), labels),
        bounds=(float(np.log(0.05)), float(np.log(20.0))),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(np.clip(np.exp(result.x), 0.05, 20.0))


def fuse_logits(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    spatial_weights: float | np.ndarray,
) -> np.ndarray:
    spatial = _validate_logits(spatial_logits)
    spectral = _validate_logits(spectral_logits)
    if spatial.shape != spectral.shape:
        raise ValueError("spatial and spectral logits must have the same shape")
    weights = np.asarray(spatial_weights, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full((len(spatial), 1), float(weights), dtype=np.float64)
    else:
        weights = weights.reshape(-1, 1)
    if len(weights) != len(spatial) or np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("spatial weights must contain one value in [0, 1] per sample")
    return weights * spatial + (1.0 - weights) * spectral


def _fixed_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in grid):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return tuple(sorted(set(grid)))


def _top_two_margin(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits)
    top_two = np.partition(probabilities, kth=-2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _global_alpha(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[float, float]:
    candidates = []
    for alpha in alpha_grid:
        loss = nll(fuse_logits(spatial_logits, spectral_logits, alpha), targets)
        candidates.append((loss, abs(alpha - 0.5), alpha))
    best_loss, _, best_alpha = min(candidates)
    return float(best_alpha), float(best_loss)


def disagreement_weights(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    *,
    global_alpha: float,
    residual_radius: float,
    margin_scale: float,
) -> np.ndarray:
    spatial = _validate_logits(spatial_logits)
    spectral = _validate_logits(spectral_logits)
    if spatial.shape != spectral.shape:
        raise ValueError("spatial and spectral logits must have the same shape")
    if not 0.0 <= global_alpha <= 1.0:
        raise ValueError("global_alpha must be in [0, 1]")
    if not 0.0 <= residual_radius <= 1.0:
        raise ValueError("residual_radius must be in [0, 1]")
    if not np.isfinite(margin_scale) or margin_scale <= 0.0:
        raise ValueError("margin_scale must be positive and finite")

    disagreement = spatial.argmax(axis=1) != spectral.argmax(axis=1)
    margin_difference = _top_two_margin(spatial) - _top_two_margin(spectral)
    residual = np.zeros(len(spatial), dtype=np.float64)
    residual[disagreement] = np.tanh(margin_difference[disagreement] / margin_scale)
    return np.clip(global_alpha + residual_radius * residual, 0.0, 1.0)


def fit_adlf_policy(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray,
    *,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    radius_grid: Iterable[float] = DEFAULT_RADIUS_GRID,
) -> ADLFPolicy:
    spatial = _validate_logits(spatial_logits, targets)
    spectral = _validate_logits(spectral_logits, targets)
    if spatial.shape != spectral.shape:
        raise ValueError("spatial and spectral logits must have the same shape")
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    alphas = _fixed_grid(alpha_grid, name="alpha_grid")
    radii = _fixed_grid(radius_grid, name="radius_grid")
    if max(radii) > 0.15:
        raise ValueError("radius_grid exceeds the pre-registered maximum radius of 0.15")

    spatial_temperature = fit_temperature(spatial, labels)
    spectral_temperature = fit_temperature(spectral, labels)
    spatial_calibrated = spatial / spatial_temperature
    spectral_calibrated = spectral / spectral_temperature
    global_alpha, global_loss = _global_alpha(
        spatial_calibrated, spectral_calibrated, labels, alphas
    )

    disagreement = spatial_calibrated.argmax(axis=1) != spectral_calibrated.argmax(axis=1)
    margin_difference = _top_two_margin(spatial_calibrated) - _top_two_margin(
        spectral_calibrated
    )
    scale_values = margin_difference[disagreement]
    if len(scale_values) < 2:
        scale_values = margin_difference
    margin_scale = max(float(np.std(scale_values)), 1e-3)

    radius_candidates = []
    for radius in radii:
        weights = disagreement_weights(
            spatial_calibrated,
            spectral_calibrated,
            global_alpha=global_alpha,
            residual_radius=radius,
            margin_scale=margin_scale,
        )
        loss = nll(fuse_logits(spatial_calibrated, spectral_calibrated, weights), labels)
        radius_candidates.append((loss, radius))
    adlf_loss, residual_radius = min(radius_candidates)

    validation_nll = {
        "spatial": nll(spatial_calibrated, labels),
        "spectral": nll(spectral_calibrated, labels),
        "mean": nll(fuse_logits(spatial_calibrated, spectral_calibrated, 0.5), labels),
        "global": global_loss,
        "adlf": float(adlf_loss),
    }
    return ADLFPolicy(
        spatial_temperature=spatial_temperature,
        spectral_temperature=spectral_temperature,
        global_alpha=global_alpha,
        residual_radius=float(residual_radius),
        margin_scale=margin_scale,
        validation_nll=validation_nll,
        alpha_grid=alphas,
        radius_grid=radii,
    )


def apply_adlf_policy(
    policy: ADLFPolicy,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    spatial = _validate_logits(spatial_logits) / policy.spatial_temperature
    spectral = _validate_logits(spectral_logits) / policy.spectral_temperature
    if spatial.shape != spectral.shape:
        raise ValueError("spatial and spectral logits must have the same shape")

    constant = lambda value: np.full(len(spatial), value, dtype=np.float64)
    global_weights = constant(policy.global_alpha)
    adlf_weights = disagreement_weights(
        spatial,
        spectral,
        global_alpha=policy.global_alpha,
        residual_radius=policy.residual_radius,
        margin_scale=policy.margin_scale,
    )
    return {
        "replay_spatial_logit_v4": (spatial, constant(1.0)),
        "replay_spectral_logit_v4": (spectral, constant(0.0)),
        "replay_mean_logit_v4": (fuse_logits(spatial, spectral, 0.5), constant(0.5)),
        "replay_global_logit_v4": (
            fuse_logits(spatial, spectral, global_weights),
            global_weights,
        ),
        "replay_adlf_v4": (fuse_logits(spatial, spectral, adlf_weights), adlf_weights),
    }
