import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from kmfm.logit_fusion import (
    apply_adlf_policy,
    disagreement_weights,
    fit_adlf_policy,
    fit_temperature,
    nll,
)
from kmfm.engine import run_experiment
from kmfm.splits import make_random_pixel_split, save_split
from scripts.aggregate import _load_runs
from scripts.replay_logit_fusion import (
    VARIANTS,
    _load_source_outputs,
    _save_variant,
    _variant_metrics,
)
from scripts.evaluate_adlf_replay import CANDIDATE, REFERENCES, evaluate_group


def test_temperature_fit_reduces_overconfident_nll():
    logits = np.array([[12.0, 0.0], [12.0, 0.0], [0.0, 12.0], [0.0, 12.0]])
    targets = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, targets)
    assert temperature > 1.0
    assert nll(logits / temperature, targets) < nll(logits, targets)


def test_disagreement_residual_is_bounded_and_inactive_on_agreement():
    spatial = np.array([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0]])
    spectral = np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    weights = disagreement_weights(
        spatial,
        spectral,
        global_alpha=0.6,
        residual_radius=0.15,
        margin_scale=0.1,
    )
    assert weights[0] == pytest.approx(0.6)
    assert np.all(weights >= 0.45)
    assert np.all(weights <= 0.75)


def test_adlf_policy_selects_positive_radius_when_confidence_identifies_branch():
    spatial = np.array(
        [[4.0, 0.0], [3.0, 0.0], [4.0, 0.0], [3.0, 0.0]], dtype=float
    )
    spectral = np.array(
        [[0.0, 3.0], [0.0, 4.0], [0.0, 3.0], [0.0, 4.0]], dtype=float
    )
    targets = np.array([0, 1, 0, 1])
    policy = fit_adlf_policy(spatial, spectral, targets)
    outputs = apply_adlf_policy(policy, spatial, spectral)
    assert policy.global_alpha == pytest.approx(0.5)
    assert policy.residual_radius > 0.0
    assert policy.validation_nll["adlf"] < policy.validation_nll["global"]
    assert set(outputs) == {
        "replay_spatial_logit_v4",
        "replay_spectral_logit_v4",
        "replay_mean_logit_v4",
        "replay_global_logit_v4",
        "replay_adlf_v4",
    }


def test_adlf_decision_applies_all_pre_registered_checks():
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "model": CANDIDATE,
                    "seed": seed,
                    "oa": 0.72,
                    "ece": 0.05,
                    "brier": 0.10,
                    "routing_auc": 0.70,
                    "selected_radius": 0.10,
                },
                {
                    "model": REFERENCES[0],
                    "seed": seed,
                    "oa": 0.70,
                    "ece": 0.05,
                    "brier": 0.11,
                    "routing_auc": 0.50,
                    "selected_radius": 0.0,
                },
                {
                    "model": REFERENCES[1],
                    "seed": seed,
                    "oa": 0.705,
                    "ece": 0.05,
                    "brier": 0.105,
                    "routing_auc": 0.50,
                    "selected_radius": 0.0,
                },
            ]
        )
    result = evaluate_group(pd.DataFrame(rows))
    assert result["decision"] == "GO"
    assert result["checks_passed"] == result["checks_total"] == 9


def test_checkpoint_replay_restores_branch_logits(tmp_path):
    rng = np.random.default_rng(9)
    labels = (np.indices((10, 10)).sum(axis=0) % 2 + 1).astype(np.int64)
    signatures = np.array([[-1.0] * 8, [1.0] * 8], dtype=np.float32)
    cube = signatures[labels - 1] + 0.05 * rng.normal(size=(10, 10, 8)).astype(np.float32)
    data_path = tmp_path / "scene.mat"
    gt_path = tmp_path / "scene_gt.mat"
    split_path = tmp_path / "split.npz"
    savemat(data_path, {"cube": cube})
    savemat(gt_path, {"gt": labels})
    save_split(
        split_path,
        make_random_pixel_split(labels, train_per_class=4, val_per_class=3, seed=3),
    )
    config = {
        "seed": 3,
        "data": {
            "name": "synthetic_replay",
            "data_path": str(data_path),
            "gt_path": str(gt_path),
            "data_key": "cube",
            "gt_key": "gt",
            "zscore_clip": 8.0,
        },
        "protocol": {"name": "random_pixel", "split_path": str(split_path)},
        "model": {
            "name": "lassf_mlp_concat_norm_v3_h64",
            "hidden_dim": 8,
            "spectral": "mlp",
            "fusion": "concat",
            "dropout": 0.0,
            "normalize_branches": True,
        },
        "training": {
            "patch_size": 5,
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 1,
            "patience": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "aux_weight": 0.2,
            "amp": False,
            "deterministic": True,
        },
        "output": {"root": str(tmp_path / "results"), "experiment": "source"},
    }
    source_run = run_experiment(config)
    loaded_config, val_result, test_result, inference_seconds, parameter_count = _load_source_outputs(
        source_run
    )
    assert loaded_config["model"]["fusion"] == "concat"
    assert val_result["spatial_logits"].shape[1] == 2
    assert test_result["spectral_logits"].shape[1] == 2
    assert inference_seconds >= 0.0
    assert parameter_count > 0

    policy = fit_adlf_policy(
        val_result["spatial_logits"],
        val_result["spectral_logits"],
        val_result["targets"],
    )
    variants = apply_adlf_policy(
        policy, test_result["spatial_logits"], test_result["spectral_logits"]
    )
    spatial_calibrated = test_result["spatial_logits"] / policy.spatial_temperature
    spectral_calibrated = test_result["spectral_logits"] / policy.spectral_temperature
    global_predictions = variants["replay_global_logit_v4"][0].argmax(axis=1)
    output_root = tmp_path / "results" / "replay"
    for variant in VARIANTS:
        logits, weights = variants[variant]
        metrics, confusion, predictions = _variant_metrics(
            variant=variant,
            seed=3,
            dataset="synthetic_replay",
            protocol="random_pixel",
            logits=logits,
            weights=weights,
            targets=test_result["targets"],
            spatial_logits=spatial_calibrated,
            spectral_logits=spectral_calibrated,
            global_predictions=global_predictions,
            policy=policy,
            inference_seconds=inference_seconds,
            parameter_count=parameter_count,
        )
        expected_alpha = {
            "replay_spatial_logit_v4": 1.0,
            "replay_spectral_logit_v4": 0.0,
            "replay_mean_logit_v4": 0.5,
            "replay_global_logit_v4": policy.global_alpha,
            "replay_adlf_v4": policy.global_alpha,
        }[variant]
        assert metrics["selected_alpha"] == pytest.approx(expected_alpha)
        _save_variant(
            run_dir=output_root / "synthetic_replay" / "random_pixel" / variant / "seed_3",
            metrics=metrics,
            confusion=confusion,
            predictions=predictions,
            logits=logits,
            weights=weights,
            targets=test_result["targets"],
            policy=policy,
            source_run=source_run,
            source_config=loaded_config,
        )
    replay_runs = _load_runs(output_root)
    assert set(replay_runs["model"]) == set(VARIANTS)
    assert replay_runs["split_sha256"].nunique() == 1
