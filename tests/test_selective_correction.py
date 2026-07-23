import numpy as np
import pandas as pd
import pytest

from kmfm.selective_correction import (
    SelectionRule,
    UtilityModel,
    _apply_rule,
    _align_final_threshold,
    apply_ssrc_policy,
    fit_ssrc_policy,
    fit_utility_model,
    select_risk_controlled_rule,
)
from scripts.aggregate import _load_runs
from scripts.evaluate_ssrc_replay import CANDIDATE, REFERENCES, evaluate_group
from scripts.replay_logit_fusion import _save_variant
from scripts.replay_selective_correction import VARIANTS, _variant_metrics


def _dummy_model() -> UtilityModel:
    return UtilityModel(
        include_class_features=False,
        num_classes=2,
        continuous_mean=(0.0, 0.0, 0.0, 0.0),
        continuous_scale=(1.0, 1.0, 1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0, 0.0),
        intercept=0.0,
        ridge_alpha=10.0,
    )


def _logits(predictions: np.ndarray, classes: int, confidence: float = 4.0) -> np.ndarray:
    values = np.zeros((len(predictions), classes), dtype=np.float64)
    values[np.arange(len(predictions)), predictions] = confidence
    return values


def test_risk_control_selects_only_statistically_safe_corrections():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    utility = np.array([1, 1, 1, -1, -1, 0])
    rule = select_risk_controlled_rule(
        scores,
        utility,
        np.ones(6, dtype=bool),
        _dummy_model(),
        coverage_grid=(0.5, 1.0),
        confidence_level=0.8,
        min_exclusive=3,
    )
    assert rule.threshold == pytest.approx(0.7)
    assert rule.oof_improved_count == 3
    assert rule.oof_harmed_count == 0
    assert rule.oof_net_corrected == 3
    assert rule.oof_wilson_lower > 0.5


def test_risk_control_abstains_when_no_candidate_is_safe():
    rule = select_risk_controlled_rule(
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([-1, 1, -1, 0]),
        np.ones(4, dtype=bool),
        _dummy_model(),
        coverage_grid=(0.5, 1.0),
        confidence_level=0.8,
        min_exclusive=2,
    )
    assert rule.threshold is None
    assert rule.target_coverage == 0.0
    assert rule.oof_selected_count == 0


def test_oof_coverage_is_exact_when_scores_are_tied():
    rule = select_risk_controlled_rule(
        np.ones(10),
        np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1]),
        np.ones(10, dtype=bool),
        _dummy_model(),
        coverage_grid=(0.5,),
        confidence_level=0.8,
        min_exclusive=5,
    )
    assert rule.oof_selected_count == 5
    assert rule.oof_improved_count == 5


def test_risk_control_rejects_partially_nonnegative_coverage():
    rule = select_risk_controlled_rule(
        np.array([0.9, 0.8, 0.7, -0.1, -0.2, -0.3]),
        np.array([1, 1, 1, 1, 1, 1]),
        np.ones(6, dtype=bool),
        _dummy_model(),
        coverage_grid=(1.0,),
        confidence_level=0.8,
        min_exclusive=3,
    )
    assert rule.threshold is None
    assert rule.target_coverage == 0.0
    assert rule.oof_selected_count == 0


def test_inference_enforces_frozen_correction_budget_under_score_shift():
    model = UtilityModel(
        include_class_features=False,
        num_classes=2,
        continuous_mean=(0.0, 0.0, 0.0, 0.0),
        continuous_scale=(1.0, 1.0, 1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0, 0.0),
        intercept=1.0,
        ridge_alpha=10.0,
    )
    rule = SelectionRule(
        model=model,
        threshold=0.5,
        target_coverage=0.25,
        oof_disagreement_count=4,
        oof_selected_count=1,
        oof_improved_count=1,
        oof_harmed_count=0,
        oof_neutral_count=0,
        oof_net_corrected=1,
        oof_wilson_lower=0.6,
        confidence_level=0.8,
        min_exclusive=1,
    )
    spatial = _logits(np.array([0, 0, 0, 0]), 2)
    spectral = _logits(np.array([1, 1, 1, 1]), 2)
    _, weights, scores = _apply_rule(rule, spatial, spectral)
    assert np.all(scores == 1.0)
    assert int((weights == 0.0).sum()) == 1


def test_final_threshold_is_recalibrated_to_final_model_score_scale():
    oof_rule = SelectionRule(
        model=_dummy_model(),
        threshold=0.8,
        target_coverage=0.5,
        oof_disagreement_count=4,
        oof_selected_count=2,
        oof_improved_count=2,
        oof_harmed_count=0,
        oof_neutral_count=0,
        oof_net_corrected=2,
        oof_wilson_lower=0.6,
        confidence_level=0.8,
        min_exclusive=2,
    )
    final_model = UtilityModel(
        include_class_features=False,
        num_classes=2,
        continuous_mean=(0.0, 0.0, 0.0, 0.0),
        continuous_scale=(1.0, 1.0, 1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0, 0.0),
        intercept=0.2,
        ridge_alpha=10.0,
    )
    spatial = _logits(np.array([0, 0, 0, 0]), 2)
    spectral = _logits(np.array([1, 1, 1, 1]), 2)
    aligned = _align_final_threshold(oof_rule, final_model, spatial, spectral)
    assert aligned.threshold == pytest.approx(0.2)
    assert aligned.model == final_model
    assert aligned.oof_net_corrected == 2


def test_class_aware_utility_model_has_bounded_feature_budget():
    targets = np.tile(np.arange(3), 4)
    spatial_prediction = np.roll(targets, 1)
    spectral_prediction = targets.copy()
    spatial = _logits(spatial_prediction, 3)
    spectral = _logits(spectral_prediction, 3)
    score_only = fit_utility_model(
        spatial, spectral, targets, include_class_features=False
    )
    class_aware = fit_utility_model(
        spatial, spectral, targets, include_class_features=True
    )
    assert len(score_only.coefficients) == 4
    assert len(class_aware.coefficients) == 4 + 2 * 3


def test_ssrc_policy_is_deterministic_and_never_switches_on_agreement():
    targets = np.repeat(np.arange(3), 10)
    spatial_prediction = targets.copy()
    spectral_prediction = targets.copy()
    spatial_prediction[[2, 3, 12, 13, 22, 23]] = np.array([1, 1, 2, 2, 0, 0])
    spectral_prediction[[0, 1, 12, 13, 20, 21]] = np.array([1, 1, 2, 2, 0, 0])
    spatial = _logits(spatial_prediction, 3, confidence=3.0)
    spectral = _logits(spectral_prediction, 3, confidence=3.5)
    first = fit_ssrc_policy(spatial, spectral, targets, seed=7, min_exclusive=2)
    second = fit_ssrc_policy(spatial, spectral, targets, seed=7, min_exclusive=2)
    assert first.to_dict() == second.to_dict()

    outputs = apply_ssrc_policy(first, spatial, spectral)
    assert set(outputs) == {
        "replay_spatial_logit_v5",
        "replay_spectral_logit_v5",
        "replay_global_logit_v5",
        "replay_ssrc_score_v5",
        "replay_ssrc_class_v5",
    }
    agreement = spatial_prediction == spectral_prediction
    for variant in ("replay_ssrc_score_v5", "replay_ssrc_class_v5"):
        logits, weights, scores = outputs[variant]
        assert logits.shape == spatial.shape
        assert scores.shape == targets.shape
        assert np.all(weights[agreement] == 1.0)
        assert set(np.unique(weights)).issubset({0.0, 1.0})


def test_ssrc_development_decision_applies_all_fixed_checks():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "model": CANDIDATE,
                "seed": seed,
                "oa": 0.73,
                "ece": 0.05,
                "brier": 0.10,
                "test_switch_count": 10,
                "oracle_recovery_vs_spatial": 0.30,
            }
        )
        for reference, oa in zip(REFERENCES, (0.70, 0.705, 0.72)):
            rows.append(
                {
                    "model": reference,
                    "seed": seed,
                    "oa": oa,
                    "ece": 0.05,
                    "brier": 0.10,
                    "test_switch_count": 0,
                    "oracle_recovery_vs_spatial": 0.0,
                }
            )
    result = evaluate_group(pd.DataFrame(rows))
    assert result["decision"] == "DEVELOPMENT_GO"
    assert result["checks_passed"] == result["checks_total"] == 13


def test_ssrc_variants_write_scores_and_aggregate_diagnostics(tmp_path):
    targets = np.repeat(np.arange(3), 10)
    spatial_prediction = targets.copy()
    spectral_prediction = targets.copy()
    spatial_prediction[[2, 3, 12, 13, 22, 23]] = np.array([1, 1, 2, 2, 0, 0])
    spectral_prediction[[0, 1, 12, 13, 20, 21]] = np.array([1, 1, 2, 2, 0, 0])
    spatial = _logits(spatial_prediction, 3, confidence=3.0)
    spectral = _logits(spectral_prediction, 3, confidence=3.5)
    policy = fit_ssrc_policy(spatial, spectral, targets, seed=4, min_exclusive=2)
    variants = apply_ssrc_policy(policy, spatial, spectral)
    spatial_calibrated = spatial / policy.spatial_temperature
    spectral_calibrated = spectral / policy.spectral_temperature
    global_predictions = variants["replay_global_logit_v5"][0].argmax(axis=1)

    source_run = tmp_path / "source"
    source_run.mkdir()
    data_path = tmp_path / "data.mat"
    gt_path = tmp_path / "gt.mat"
    split_path = tmp_path / "split.npz"
    for path in (data_path, gt_path, split_path, source_run / "checkpoint_best.pt"):
        path.write_bytes(b"fixture")
    (source_run / "resolved_config.json").write_text("{}", encoding="utf-8")
    source_config = {
        "model": {"name": "source_concat"},
        "protocol": {"split_path": str(split_path)},
        "data": {"data_path": str(data_path), "gt_path": str(gt_path)},
    }
    output_root = tmp_path / "results"
    for variant in VARIANTS:
        logits, weights, scores = variants[variant]
        metrics, confusion, predictions = _variant_metrics(
            variant=variant,
            seed=4,
            dataset="synthetic",
            protocol="spatial_block",
            logits=logits,
            weights=weights,
            selector_scores=scores,
            targets=targets,
            spatial_logits=spatial_calibrated,
            spectral_logits=spectral_calibrated,
            global_predictions=global_predictions,
            policy=policy,
            source_inference_seconds=0.1,
            selector_policy_seconds=0.01,
            source_parameter_count=100,
        )
        run_dir = output_root / "synthetic" / "spatial_block" / variant / "seed_4"
        _save_variant(
            run_dir=run_dir,
            metrics=metrics,
            confusion=confusion,
            predictions=predictions,
            logits=logits,
            weights=weights,
            targets=targets,
            policy=policy,
            source_run=source_run,
            source_config=source_config,
            extra_arrays=(
                {"selector_scores.npy": scores.astype(np.float32)}
                if scores is not None
                else None
            ),
        )
    runs = _load_runs(output_root)
    assert set(runs["model"]) == set(VARIANTS)
    candidate = runs[runs["model"] == CANDIDATE].iloc[0]
    assert candidate["selector_parameter_count"] == 4 + 2 * 3 + 1
    candidate_dir = output_root / "synthetic" / "spatial_block" / CANDIDATE / "seed_4"
    assert (candidate_dir / "selector_scores.npy").is_file()
    manifest = (candidate_dir / "manifest.json").read_text(encoding="utf-8")
    assert "selector_scores.npy" in manifest
