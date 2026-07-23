import json
import sys

import numpy as np
import pandas as pd
import pytest

from kmfm.complementarity import analyze_seed, summarize_class_rows
from scripts.audit_branch_complementarity import VARIANTS, main as audit_main


def _write_run(root, variant, predictions, targets):
    run_dir = (
        root
        / "results"
        / "audit_source"
        / "synthetic"
        / "spatial_block"
        / variant
        / "seed_0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"state": "success", "seed": 0}), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "selected_alpha": 0.6,
                "selected_radius": 0.1,
                "routing_auc": 0.55,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"inputs": {"split": {"sha256": "fixed-split"}}}),
        encoding="utf-8",
    )
    np.save(run_dir / "test_targets.npy", targets)
    np.save(run_dir / "test_predictions.npy", predictions)


def test_complementarity_audit_separates_oracle_and_realized_corrections():
    targets = np.array([0, 0, 1, 1, 2, 2, 2])
    spatial = np.array([0, 1, 1, 0, 2, 0, 1])
    spectral = np.array([1, 0, 1, 1, 0, 2, 1])
    global_predictions = np.array([0, 1, 1, 1, 2, 0, 1])
    adlf = np.array([0, 1, 1, 1, 2, 0, 1])
    seed, rows = analyze_seed(
        targets,
        spatial,
        spectral,
        global_predictions=global_predictions,
        adlf_predictions=adlf,
        selected_alpha=0.5,
        selected_radius=0.1,
    )
    assert seed["spatial_oa"] == pytest.approx(3 / 7)
    assert seed["spectral_oa"] == pytest.approx(4 / 7)
    assert seed["oracle_oa"] == pytest.approx(6 / 7)
    assert seed["exclusive_spectral_correct"] == 3
    assert seed["exclusive_spatial_correct"] == 2
    assert seed["global_improved_vs_spatial"] == 1
    assert seed["global_harmed_vs_spatial"] == 0
    class_summary = summarize_class_rows(rows)
    assert len(class_summary) == 3
    class_zero = next(row for row in class_summary if row["class_index"] == 0)
    assert class_zero["support_mean"] == pytest.approx(2.0)
    assert class_zero["exclusive_spectral_correct_mean"] == pytest.approx(1.0)


def test_complementarity_audit_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="same number of samples"):
        analyze_seed(
            np.array([0, 1]),
            np.array([0]),
            np.array([0, 1]),
        )


def test_complementarity_audit_cli_writes_traceable_reports(tmp_path, monkeypatch):
    targets = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)
    predictions = {
        "spatial": np.array([0, 1, 1, 0, 2, 0], dtype=np.int16),
        "spectral": np.array([1, 0, 1, 1, 0, 2], dtype=np.int16),
        "global": np.array([0, 1, 1, 1, 2, 0], dtype=np.int16),
        "adlf": np.array([0, 1, 1, 1, 2, 0], dtype=np.int16),
    }
    for name, variant in VARIANTS.items():
        _write_run(tmp_path, variant, predictions[name], targets)

    output_dir = tmp_path / "audit-report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_branch_complementarity.py",
            "--project-root",
            str(tmp_path),
            "--experiment",
            "audit_source",
            "--dataset",
            "synthetic",
            "--protocol",
            "spatial_block",
            "--seeds",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )
    audit_main()

    expected = {
        "complementarity_per_seed.csv",
        "complementarity_per_class_seed.csv",
        "complementarity_per_class_summary.csv",
        "complementarity_audit.json",
        "complementarity_audit.md",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    seed_frame = pd.read_csv(output_dir / "complementarity_per_seed.csv")
    assert seed_frame.loc[0, "oracle_oa"] == 1.0
    assert seed_frame.loc[0, "exclusive_spectral_correct"] == 3
    payload = json.loads(
        (output_dir / "complementarity_audit.json").read_text(encoding="utf-8")
    )
    assert payload["decision"]["complementarity_decision"] == "CONTINUE_SELECTOR_DEVELOPMENT"
    assert len(payload["sources"]) == 4
