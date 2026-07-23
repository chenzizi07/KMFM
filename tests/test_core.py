import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

torch = pytest.importorskip("torch")

from kmfm.data import HSIPatchDataset
from kmfm.engine import _fit_temperature, run_experiment
from kmfm.metrics import classification_metrics, probabilistic_metrics, routing_diagnostics
from kmfm.model import FusionModule, LASSFNet, SpectralConv1D
from kmfm.splits import TRAIN, VAL, make_random_pixel_split, save_split
from scripts.aggregate import _load_runs, _summary
from scripts.evaluate_mechanism import CANDIDATE, REFERENCES, evaluate_group


def test_metrics_are_from_one_confusion_matrix():
    metrics, confusion = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), num_classes=2
    )
    assert confusion.tolist() == [[1, 1], [0, 2]]
    assert metrics["oa"] == pytest.approx(0.75)
    assert metrics["aa"] == pytest.approx(0.75)


def test_random_split_is_disjoint_and_fixed_count():
    labels = np.tile(np.arange(1, 4), (20, 20))[:, :20]
    split = make_random_pixel_split(labels, train_per_class=5, val_per_class=3, seed=9)
    assert not np.any(split.train_mask & split.val_mask)
    assert not np.any(split.train_mask & split.test_mask)
    assert split.metadata["counts"]["train"] == [5, 5, 5]
    assert split.metadata["counts"]["val"] == [3, 3, 3]


def test_context_guard_zeros_other_regions():
    cube = np.ones((5, 5, 4), dtype=np.float32)
    labels = np.ones((5, 5), dtype=np.int64)
    centers = np.zeros((5, 5), dtype=bool)
    centers[2, 2] = True
    regions = np.full((5, 5), VAL, dtype=np.int8)
    regions[1:4, 1:4] = TRAIN
    dataset = HSIPatchDataset(
        cube,
        labels,
        centers,
        patch_size=5,
        region_map=regions,
        region_value=TRAIN,
        allow_full_context=False,
    )
    patch, _, _, visible = dataset[0]
    assert visible.sum().item() == 9
    assert patch[:, 0, 0].abs().sum().item() == 0
    assert patch[:, 2, 2].sum().item() == 4


def test_spectral_conv_uses_band_length():
    encoder = SpectralConv1D(hidden_dim=8, kernels=(3, 5), branch_dim=4)
    spectrum = torch.randn(2, 17, requires_grad=True)
    output = encoder(spectrum)
    assert output.shape == (2, 8)
    output.sum().backward()
    assert spectrum.grad is not None


def test_model_forward_backward():
    model = LASSFNet(bands=12, num_classes=3, hidden_dim=16)
    patch = torch.randn(2, 12, 7, 7)
    visible = torch.ones(2, 7, 7)
    output = model(patch, visible)
    assert output["logits"].shape == (2, 3)
    loss = output["logits"].sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_entropy_softmax_prefers_lower_entropy_without_entropy_gradient():
    fusion = FusionModule(hidden_dim=4, mode="entropy_softmax", entropy_temperature=0.25)
    spatial = torch.ones(2, 4, requires_grad=True)
    spectral = torch.zeros(2, 4, requires_grad=True)
    spatial_entropy = torch.tensor([[0.1], [0.9]], requires_grad=True)
    spectral_entropy = torch.tensor([[0.9], [0.1]], requires_grad=True)
    fused, gate = fusion(spatial, spectral, spatial_entropy, spectral_entropy)
    assert gate[0].item() > 0.9
    assert gate[1].item() < 0.1
    fused.sum().backward()
    assert spatial.grad is not None
    assert spectral.grad is not None
    assert spatial_entropy.grad is None
    assert spectral_entropy.grad is None


def test_normalized_model_reports_comparable_feature_norms():
    model = LASSFNet(
        bands=12,
        num_classes=3,
        hidden_dim=16,
        spectral="mlp",
        fusion="entropy_softmax",
        normalize_branches=True,
    )
    output = model(torch.randn(3, 12, 7, 7), torch.ones(3, 7, 7))
    expected_norm = 16**0.5
    assert output["spatial_feature_norm"].mean().item() == pytest.approx(expected_norm, rel=0.02)
    assert output["spectral_feature_norm"].mean().item() == pytest.approx(expected_norm, rel=0.02)
    assert torch.all((output["contribution_ratio"] >= 0) & (output["contribution_ratio"] <= 1))


def test_probabilistic_and_routing_metrics():
    targets = np.array([0, 1, 0, 1])
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 0.0], [0.0, 2.0]])
    probability = probabilistic_metrics(targets, logits, num_bins=4)
    assert probability["nll"] < 0.1
    assert probability["brier"] < 0.05

    routing = routing_diagnostics(
        targets,
        fused_predictions=targets,
        spatial_predictions=np.array([0, 0, 0, 0]),
        spectral_predictions=np.array([1, 1, 1, 1]),
        spatial_weights=np.array([0.9, 0.1, 0.8, 0.2]),
    )
    assert routing["routing_auc"] == pytest.approx(1.0)
    assert routing["oracle_oa"] == pytest.approx(1.0)


def test_temperature_fit_reduces_overconfidence():
    logits = np.array([[12.0, 0.0], [12.0, 0.0], [0.0, 12.0], [0.0, 12.0]])
    targets = np.array([0, 1, 1, 0])
    temperature = _fit_temperature(logits, targets)
    assert temperature > 1.0


def test_entropy_v3_run_writes_calibration_and_diagnostics(tmp_path):
    rng = np.random.default_rng(4)
    labels = (np.indices((10, 10)).sum(axis=0) % 2 + 1).astype(np.int64)
    signatures = np.array(
        [[-1.0] * 8, [1.0] * 8], dtype=np.float32
    )
    cube = signatures[labels - 1] + 0.1 * rng.normal(size=(10, 10, 8)).astype(np.float32)
    data_path = tmp_path / "scene.mat"
    gt_path = tmp_path / "scene_gt.mat"
    split_path = tmp_path / "split.npz"
    savemat(data_path, {"cube": cube})
    savemat(gt_path, {"gt": labels})
    save_split(
        split_path,
        make_random_pixel_split(labels, train_per_class=4, val_per_class=3, seed=2),
    )
    config = {
        "seed": 2,
        "data": {
            "name": "synthetic",
            "data_path": str(data_path),
            "gt_path": str(gt_path),
            "data_key": "cube",
            "gt_key": "gt",
            "zscore_clip": 8.0,
        },
        "protocol": {"name": "random_pixel", "split_path": str(split_path)},
        "model": {
            "name": "entropy_v3_test",
            "hidden_dim": 8,
            "spectral": "mlp",
            "fusion": "entropy_softmax",
            "dropout": 0.0,
            "normalize_branches": True,
            "entropy_temperature": 0.25,
            "calibrate_branch_temperatures": True,
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
        "output": {"root": str(tmp_path / "results"), "experiment": "integration"},
    }
    run_dir = run_experiment(config)
    assert (run_dir / "calibration.json").is_file()
    assert (run_dir / "contribution_ratio.npy").is_file()
    assert (run_dir / "spatial_predictions.npy").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert metrics["nll"] >= 0
    assert metrics["routing_auc"] is None or 0 <= metrics["routing_auc"] <= 1
    assert calibration["spatial_nll_after"] <= calibration["spatial_nll_before"] + 1e-8
    assert calibration["spectral_nll_after"] <= calibration["spectral_nll_before"] + 1e-8
    runs = _load_runs(tmp_path / "results" / "integration")
    summary = _summary(runs)
    assert summary.loc[0, "nll_mean"] == pytest.approx(metrics["nll"])


def test_mechanism_decision_applies_pre_registered_thresholds():
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "model": CANDIDATE,
                    "seed": seed,
                    "oa": 0.72,
                    "ece": 0.04,
                    "brier": 0.10,
                    "routing_auc": 0.70,
                },
                {
                    "model": REFERENCES[0],
                    "seed": seed,
                    "oa": 0.70,
                    "ece": 0.06,
                    "brier": 0.12,
                    "routing_auc": np.nan,
                },
                {
                    "model": REFERENCES[1],
                    "seed": seed,
                    "oa": 0.705,
                    "ece": 0.05,
                    "brier": 0.11,
                    "routing_auc": 0.50,
                },
            ]
        )
    result = evaluate_group(pd.DataFrame(rows))
    assert result["decision"] == "GO"
    assert result["checks_passed"] == result["checks_total"]
