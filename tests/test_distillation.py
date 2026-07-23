import json

import numpy as np
import pytest
import pandas as pd
from scipy.io import savemat

torch = pytest.importorskip("torch")

from kmfm.distillation import (
    advantage_weighted_distillation_loss,
    class_advantage_profile,
    stratified_folds,
)
from kmfm.engine import run_experiment
from kmfm.splits import make_random_pixel_split, save_split
from scripts.evaluate_oasd import BASELINE, CANDIDATE, UNIFORM, evaluate_group


def test_stratified_folds_are_deterministic_and_exhaustive():
    targets = np.repeat(np.arange(3), 6)
    first = stratified_folds(targets, n_splits=3, seed=11)
    second = stratified_folds(targets, n_splits=3, seed=11)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(len(targets)))
    for fold in first:
        assert np.bincount(targets[fold], minlength=3).tolist() == [2, 2, 2]


def test_class_advantage_only_weights_spectral_positive_classes():
    targets = np.repeat(np.arange(2), 6)
    spatial = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0])
    spectral = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    profile = class_advantage_profile(
        targets,
        spatial,
        spectral,
        num_classes=2,
        prior_strength=0.0,
        reference_gain=0.5,
    )
    assert profile.class_weights[0] > 0
    assert profile.class_weights[1] == 0
    assert profile.per_class[0]["spectral_only_correct"] > profile.per_class[0]["spatial_only_correct"]


def test_distillation_detaches_teacher_and_respects_zero_weights():
    student = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    zero, mean_weight = advantage_weighted_distillation_loss(
        student, teacher, targets, torch.zeros(2)
    )
    assert zero.item() == pytest.approx(0.0)
    assert mean_weight.item() == pytest.approx(0.0)

    loss, _ = advantage_weighted_distillation_loss(
        student, teacher, targets, torch.ones(2)
    )
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_oasd_decision_applies_all_fixed_checks():
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "model": CANDIDATE,
                    "seed": seed,
                    "oa": 0.72,
                    "aa": 0.71,
                    "ece": 0.04,
                    "brier": 0.10,
                    "distillation_active_class_count": 2,
                },
                {
                    "model": BASELINE,
                    "seed": seed,
                    "oa": 0.71,
                    "aa": 0.70,
                    "ece": 0.05,
                    "brier": 0.11,
                    "distillation_active_class_count": 0,
                },
                {
                    "model": UNIFORM,
                    "seed": seed,
                    "oa": 0.715,
                    "aa": 0.705,
                    "ece": 0.045,
                    "brier": 0.105,
                    "distillation_active_class_count": 2,
                },
            ]
        )
    decision = evaluate_group(pd.DataFrame(rows))
    assert decision["decision"] == "DEVELOPMENT_GO"
    assert decision["checks_passed"] == decision["checks_total"]


def test_oof_distillation_run_writes_profile(tmp_path):
    rng = np.random.default_rng(8)
    labels = (np.indices((8, 8)).sum(axis=0) % 2 + 1).astype(np.int64)
    signatures = np.array([[-1.0] * 6, [1.0] * 6], dtype=np.float32)
    cube = signatures[labels - 1] + 0.1 * rng.normal(size=(8, 8, 6)).astype(np.float32)
    data_path = tmp_path / "scene.mat"
    gt_path = tmp_path / "scene_gt.mat"
    split_path = tmp_path / "split.npz"
    savemat(data_path, {"cube": cube})
    savemat(gt_path, {"gt": labels})
    save_split(
        split_path,
        make_random_pixel_split(labels, train_per_class=4, val_per_class=2, seed=3),
    )
    config = {
        "seed": 3,
        "data": {
            "name": "synthetic_oasd",
            "data_path": str(data_path),
            "gt_path": str(gt_path),
            "data_key": "cube",
            "gt_key": "gt",
            "zscore_clip": 8.0,
        },
        "protocol": {"name": "random_pixel", "split_path": str(split_path)},
        "model": {
            "name": "oasd_test",
            "hidden_dim": 8,
            "spectral": "mlp",
            "fusion": "spatial_only",
            "dropout": 0.0,
            "normalize_branches": True,
            "distillation": {
                "mode": "oof_class",
                "coefficient": 0.5,
                "temperature": 2.0,
                "folds": 2,
                "oof_epochs": 1,
                "oof_aux_weight": 0.5,
            },
        },
        "training": {
            "patch_size": 3,
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
        "output": {"root": str(tmp_path / "results"), "experiment": "integration"},
    }
    run_dir = run_experiment(config)
    profile = json.loads(
        (run_dir / "distillation_profile.json").read_text(encoding="utf-8")
    )
    assert profile["mode"] == "oof_class"
    assert profile["folds"] == 2
    assert len(profile["class_weights"]) == 2
    assert (run_dir / "metrics.json").is_file()
