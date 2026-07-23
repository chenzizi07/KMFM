from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from .logit_fusion import fit_temperature, fuse_logits, nll, softmax


DEFAULT_ALPHA_GRID = tuple(np.linspace(0.0, 1.0, 21).tolist())
DEFAULT_COVERAGE_GRID = (0.05, 0.10, 0.20, 0.30)
CONTINUOUS_FEATURE_NAMES = (
    "spectral_minus_spatial_confidence",
    "spectral_minus_spatial_margin",
    "spatial_minus_spectral_entropy",
    "jensen_shannon_divergence",
)


@dataclass(frozen=True)
class UtilityModel:
    include_class_features: bool
    num_classes: int
    continuous_mean: tuple[float, ...]
    continuous_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    ridge_alpha: float


@dataclass(frozen=True)
class SelectionRule:
    model: UtilityModel
    threshold: float | None
    target_coverage: float
    oof_disagreement_count: int
    oof_selected_count: int
    oof_improved_count: int
    oof_harmed_count: int
    oof_neutral_count: int
    oof_net_corrected: int
    oof_wilson_lower: float | None
    confidence_level: float
    min_exclusive: int


@dataclass(frozen=True)
class SSRCPolicy:
    spatial_temperature: float
    spectral_temperature: float
    global_alpha: float
    score_only: SelectionRule
    class_aware: SelectionRule
    validation_nll: dict[str, float]
    alpha_grid: tuple[float, ...]
    coverage_grid: tuple[float, ...]
    folds: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    spatial = np.asarray(spatial_logits, dtype=np.float64)
    spectral = np.asarray(spectral_logits, dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("spatial_logits must have shape (samples, classes)")
    if spatial.shape != spectral.shape:
        raise ValueError("spatial and spectral logits must have the same shape")
    if not np.isfinite(spatial).all() or not np.isfinite(spectral).all():
        raise ValueError("branch logits contain NaN or infinite values")
    labels = None
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        if len(labels) != len(spatial):
            raise ValueError("targets and logits must contain the same number of samples")
        if np.any((labels < 0) | (labels >= spatial.shape[1])):
            raise ValueError("targets contain an invalid class index")
    return spatial, spectral, labels


def _margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, kth=-2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    normalizer = np.log(probabilities.shape[1])
    return (
        -(probabilities * np.log(probabilities.clip(1e-12, 1.0))).sum(axis=1)
        / normalizer
    )


def _continuous_features(
    spatial_logits: np.ndarray, spectral_logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_probability = softmax(spatial_logits)
    spectral_probability = softmax(spectral_logits)
    spatial_prediction = spatial_probability.argmax(axis=1)
    spectral_prediction = spectral_probability.argmax(axis=1)
    mean_probability = 0.5 * (spatial_probability + spectral_probability)
    js_divergence = 0.5 * (
        (
            spatial_probability
            * np.log(
                spatial_probability.clip(1e-12, 1.0)
                / mean_probability.clip(1e-12, 1.0)
            )
        ).sum(axis=1)
        + (
            spectral_probability
            * np.log(
                spectral_probability.clip(1e-12, 1.0)
                / mean_probability.clip(1e-12, 1.0)
            )
        ).sum(axis=1)
    )
    features = np.column_stack(
        [
            spectral_probability.max(axis=1) - spatial_probability.max(axis=1),
            _margin(spectral_probability) - _margin(spatial_probability),
            _entropy(spatial_probability) - _entropy(spectral_probability),
            js_divergence,
        ]
    )
    return features, spatial_prediction, spectral_prediction


def _design_matrix(
    continuous: np.ndarray,
    spatial_prediction: np.ndarray,
    spectral_prediction: np.ndarray,
    *,
    include_class_features: bool,
    num_classes: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (continuous - mean) / scale
    if not include_class_features:
        return standardized
    identity = np.eye(num_classes, dtype=np.float64)
    return np.column_stack(
        [standardized, identity[spatial_prediction], identity[spectral_prediction]]
    )


def _utility(
    targets: np.ndarray,
    spatial_prediction: np.ndarray,
    spectral_prediction: np.ndarray,
) -> np.ndarray:
    return (
        (spectral_prediction == targets).astype(np.int8)
        - (spatial_prediction == targets).astype(np.int8)
    )


def fit_utility_model(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray,
    *,
    include_class_features: bool,
    ridge_alpha: float = 10.0,
) -> UtilityModel:
    spatial, spectral, labels = _validate(spatial_logits, spectral_logits, targets)
    assert labels is not None
    if ridge_alpha <= 0 or not np.isfinite(ridge_alpha):
        raise ValueError("ridge_alpha must be positive and finite")
    continuous, spatial_prediction, spectral_prediction = _continuous_features(
        spatial, spectral
    )
    disagreement = spatial_prediction != spectral_prediction
    training_continuous = continuous[disagreement]
    if not len(training_continuous):
        training_continuous = continuous
    mean = training_continuous.mean(axis=0)
    scale = training_continuous.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    design = _design_matrix(
        continuous,
        spatial_prediction,
        spectral_prediction,
        include_class_features=include_class_features,
        num_classes=spatial.shape[1],
        mean=mean,
        scale=scale,
    )
    utility = _utility(labels, spatial_prediction, spectral_prediction)
    if not np.any(disagreement):
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        intercept = 0.0
    else:
        estimator = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        estimator.fit(design[disagreement], utility[disagreement])
        coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        intercept = float(estimator.intercept_)
    return UtilityModel(
        include_class_features=include_class_features,
        num_classes=spatial.shape[1],
        continuous_mean=tuple(float(value) for value in mean),
        continuous_scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
        ridge_alpha=float(ridge_alpha),
    )


def utility_scores(
    model: UtilityModel,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
) -> np.ndarray:
    spatial, spectral, _ = _validate(spatial_logits, spectral_logits)
    if spatial.shape[1] != model.num_classes:
        raise ValueError("selector class count does not match branch logits")
    continuous, spatial_prediction, spectral_prediction = _continuous_features(
        spatial, spectral
    )
    design = _design_matrix(
        continuous,
        spatial_prediction,
        spectral_prediction,
        include_class_features=model.include_class_features,
        num_classes=model.num_classes,
        mean=np.asarray(model.continuous_mean, dtype=np.float64),
        scale=np.asarray(model.continuous_scale, dtype=np.float64),
    )
    return design @ np.asarray(model.coefficients, dtype=np.float64) + model.intercept


def _wilson_lower(successes: int, trials: int, confidence_level: float) -> float | None:
    if trials <= 0:
        return None
    z = float(stats.norm.ppf(confidence_level))
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return float((center - radius) / denominator)


def select_risk_controlled_rule(
    scores: np.ndarray,
    utility: np.ndarray,
    disagreement: np.ndarray,
    model: UtilityModel,
    *,
    coverage_grid: Iterable[float] = DEFAULT_COVERAGE_GRID,
    confidence_level: float = 0.80,
    min_exclusive: int = 5,
) -> SelectionRule:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    outcomes = np.asarray(utility, dtype=np.int8).reshape(-1)
    disagrees = np.asarray(disagreement, dtype=bool).reshape(-1)
    if not (values.shape == outcomes.shape == disagrees.shape):
        raise ValueError("scores, utility, and disagreement must have the same shape")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1.0)")
    if min_exclusive < 1:
        raise ValueError("min_exclusive must be positive")
    grid = tuple(sorted(set(float(value) for value in coverage_grid)))
    if not grid or any(value <= 0.0 or value > 1.0 for value in grid):
        raise ValueError("coverage_grid must contain values in (0, 1]")
    available = np.flatnonzero(disagrees & np.isfinite(values))
    safe_candidates: list[dict[str, float | int]] = []
    for coverage in grid:
        count = max(1, int(np.ceil(coverage * len(available)))) if len(available) else 0
        threshold = None
        selected = np.zeros(len(values), dtype=bool)
        if count:
            ranked = available[np.argsort(-values[available], kind="mergesort")]
            chosen = ranked[: min(count, len(ranked))]
            # A fixed-coverage candidate is evaluated as a whole. Silently
            # dropping negative-score members would understate its OOF risk
            # while retaining the larger inference-time coverage budget.
            if len(chosen) == count and values[chosen[-1]] >= 0.0:
                threshold = float(values[chosen[-1]])
                selected[chosen] = True
        improved = int(np.sum(selected & (outcomes == 1)))
        harmed = int(np.sum(selected & (outcomes == -1)))
        neutral = int(np.sum(selected & (outcomes == 0)))
        exclusive = improved + harmed
        lower = _wilson_lower(improved, exclusive, confidence_level)
        net = improved - harmed
        if (
            exclusive >= min_exclusive
            and net > 0
            and lower is not None
            and lower > 0.5
            and threshold is not None
        ):
            details: dict[str, float | int] = {
                "coverage": coverage,
                "threshold": threshold,
                "selected": int(selected.sum()),
                "improved": improved,
                "harmed": harmed,
                "neutral": neutral,
                "net": net,
                "lower": lower,
            }
            safe_candidates.append(details)
    if safe_candidates:
        details = max(
            safe_candidates,
            key=lambda item: (
                int(item["net"]),
                float(item["lower"]),
                -int(item["selected"]),
                -float(item["coverage"]),
                float(item["threshold"]),
            ),
        )
        threshold = float(details["threshold"])
        target_coverage = float(details["coverage"])
        selected_count = int(details["selected"])
        improved = int(details["improved"])
        harmed = int(details["harmed"])
        neutral = int(details["neutral"])
        lower = float(details["lower"])
    else:
        threshold = None
        target_coverage = 0.0
        selected_count = improved = harmed = neutral = 0
        lower = None
    return SelectionRule(
        model=model,
        threshold=threshold,
        target_coverage=target_coverage,
        oof_disagreement_count=int(len(available)),
        oof_selected_count=selected_count,
        oof_improved_count=improved,
        oof_harmed_count=harmed,
        oof_neutral_count=neutral,
        oof_net_corrected=improved - harmed,
        oof_wilson_lower=lower,
        confidence_level=float(confidence_level),
        min_exclusive=int(min_exclusive),
    )


def _global_alpha(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[float, float]:
    candidates = [
        (
            nll(fuse_logits(spatial_logits, spectral_logits, alpha), targets),
            abs(alpha - 0.5),
            alpha,
        )
        for alpha in alpha_grid
    ]
    loss, _, alpha = min(candidates)
    return float(alpha), float(loss)


def _align_final_threshold(
    rule: SelectionRule,
    final_model: UtilityModel,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
) -> SelectionRule:
    if rule.target_coverage <= 0.0:
        return SelectionRule(
            **{
                **asdict(rule),
                "model": final_model,
                "threshold": None,
            }
        )
    scores = utility_scores(final_model, spatial_logits, spectral_logits)
    disagreement = spatial_logits.argmax(axis=1) != spectral_logits.argmax(axis=1)
    available = np.flatnonzero(disagreement & np.isfinite(scores))
    count = max(1, int(np.ceil(rule.target_coverage * len(available)))) if len(available) else 0
    threshold = None
    if count:
        ranked = available[np.argsort(-scores[available], kind="mergesort")]
        threshold = max(0.0, float(scores[ranked[min(count, len(ranked)) - 1]]))
    return SelectionRule(
        model=final_model,
        threshold=threshold,
        target_coverage=rule.target_coverage if threshold is not None else 0.0,
        oof_disagreement_count=rule.oof_disagreement_count,
        oof_selected_count=rule.oof_selected_count,
        oof_improved_count=rule.oof_improved_count,
        oof_harmed_count=rule.oof_harmed_count,
        oof_neutral_count=rule.oof_neutral_count,
        oof_net_corrected=rule.oof_net_corrected,
        oof_wilson_lower=rule.oof_wilson_lower,
        confidence_level=rule.confidence_level,
        min_exclusive=rule.min_exclusive,
    )


def fit_ssrc_policy(
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    ridge_alpha: float = 10.0,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    coverage_grid: Iterable[float] = DEFAULT_COVERAGE_GRID,
    confidence_level: float = 0.80,
    min_exclusive: int = 5,
) -> SSRCPolicy:
    spatial, spectral, labels = _validate(spatial_logits, spectral_logits, targets)
    assert labels is not None
    alphas = tuple(sorted(set(float(value) for value in alpha_grid)))
    coverages = tuple(sorted(set(float(value) for value in coverage_grid)))
    if not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("alpha_grid must contain values in [0, 1]")
    class_counts = np.bincount(labels, minlength=spatial.shape[1])
    positive_counts = class_counts[class_counts > 0]
    actual_folds = min(int(folds), int(positive_counts.min()))
    if actual_folds < 2:
        raise ValueError("at least two samples per represented class are required")

    spatial_prediction = spatial.argmax(axis=1)
    spectral_prediction = spectral.argmax(axis=1)
    disagreement = spatial_prediction != spectral_prediction
    utility = _utility(labels, spatial_prediction, spectral_prediction)
    oof_scores = {
        False: np.full(len(labels), np.nan, dtype=np.float64),
        True: np.full(len(labels), np.nan, dtype=np.float64),
    }
    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=int(seed))
    for train_index, holdout_index in splitter.split(np.zeros(len(labels)), labels):
        spatial_temperature = fit_temperature(spatial[train_index], labels[train_index])
        spectral_temperature = fit_temperature(spectral[train_index], labels[train_index])
        spatial_train = spatial[train_index] / spatial_temperature
        spectral_train = spectral[train_index] / spectral_temperature
        spatial_holdout = spatial[holdout_index] / spatial_temperature
        spectral_holdout = spectral[holdout_index] / spectral_temperature
        for include_classes in (False, True):
            model = fit_utility_model(
                spatial_train,
                spectral_train,
                labels[train_index],
                include_class_features=include_classes,
                ridge_alpha=ridge_alpha,
            )
            oof_scores[include_classes][holdout_index] = utility_scores(
                model, spatial_holdout, spectral_holdout
            )

    final_spatial_temperature = fit_temperature(spatial, labels)
    final_spectral_temperature = fit_temperature(spectral, labels)
    spatial_calibrated = spatial / final_spatial_temperature
    spectral_calibrated = spectral / final_spectral_temperature
    global_alpha, global_loss = _global_alpha(
        spatial_calibrated, spectral_calibrated, labels, alphas
    )
    rules: dict[bool, SelectionRule] = {}
    for include_classes in (False, True):
        final_model = fit_utility_model(
            spatial_calibrated,
            spectral_calibrated,
            labels,
            include_class_features=include_classes,
            ridge_alpha=ridge_alpha,
        )
        oof_rule = select_risk_controlled_rule(
            oof_scores[include_classes],
            utility,
            disagreement,
            final_model,
            coverage_grid=coverages,
            confidence_level=confidence_level,
            min_exclusive=min_exclusive,
        )
        rules[include_classes] = _align_final_threshold(
            oof_rule,
            final_model,
            spatial_calibrated,
            spectral_calibrated,
        )
    return SSRCPolicy(
        spatial_temperature=final_spatial_temperature,
        spectral_temperature=final_spectral_temperature,
        global_alpha=global_alpha,
        score_only=rules[False],
        class_aware=rules[True],
        validation_nll={
            "spatial": nll(spatial_calibrated, labels),
            "spectral": nll(spectral_calibrated, labels),
            "global": global_loss,
        },
        alpha_grid=alphas,
        coverage_grid=coverages,
        folds=actual_folds,
        seed=int(seed),
    )


def _apply_rule(
    rule: SelectionRule,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_prediction = spatial_logits.argmax(axis=1)
    spectral_prediction = spectral_logits.argmax(axis=1)
    disagreement = spatial_prediction != spectral_prediction
    scores = utility_scores(rule.model, spatial_logits, spectral_logits)
    selected = np.zeros(len(spatial_logits), dtype=bool)
    if rule.threshold is not None:
        eligible = np.flatnonzero(disagreement & (scores >= rule.threshold))
        correction_budget = int(
            np.ceil(rule.target_coverage * int(disagreement.sum()))
        )
        if correction_budget > 0 and len(eligible):
            ranked = eligible[np.argsort(-scores[eligible], kind="mergesort")]
            selected[ranked[:correction_budget]] = True
    logits = np.where(selected[:, None], spectral_logits, spatial_logits)
    spatial_weights = np.where(selected, 0.0, 1.0)
    return logits, spatial_weights, scores


def apply_ssrc_policy(
    policy: SSRCPolicy,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    spatial, spectral, _ = _validate(spatial_logits, spectral_logits)
    spatial = spatial / policy.spatial_temperature
    spectral = spectral / policy.spectral_temperature
    count = len(spatial)
    global_weights = np.full(count, policy.global_alpha, dtype=np.float64)
    score_logits, score_weights, score_values = _apply_rule(
        policy.score_only, spatial, spectral
    )
    class_logits, class_weights, class_values = _apply_rule(
        policy.class_aware, spatial, spectral
    )
    return {
        "replay_spatial_logit_v5": (
            spatial,
            np.ones(count, dtype=np.float64),
            None,
        ),
        "replay_spectral_logit_v5": (
            spectral,
            np.zeros(count, dtype=np.float64),
            None,
        ),
        "replay_global_logit_v5": (
            fuse_logits(spatial, spectral, global_weights),
            global_weights,
            None,
        ),
        "replay_ssrc_score_v5": (score_logits, score_weights, score_values),
        "replay_ssrc_class_v5": (class_logits, class_weights, class_values),
    }
