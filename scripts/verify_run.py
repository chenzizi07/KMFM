from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kmfm.metrics import classification_metrics, probabilistic_metrics, routing_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute a run from saved prediction and labels")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    prediction = np.load(run_dir / "prediction.npy")
    target = np.load(run_dir / "ground_truth_eval.npy")
    saved_confusion = np.load(run_dir / "confusion_matrix.npy")
    valid = target >= 0
    num_classes = saved_confusion.shape[0]
    metrics, confusion = classification_metrics(target[valid], prediction[valid], num_classes)
    if not np.array_equal(confusion, saved_confusion):
        raise SystemExit("FAILED: saved confusion matrix does not match prediction/ground truth")
    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for name in ("oa", "aa", "kappa"):
        if not np.isclose(metrics[name], saved_metrics[name], rtol=0, atol=1e-12, equal_nan=True):
            raise SystemExit(
                f"FAILED: {name} recomputed as {metrics[name]} but metrics.json has {saved_metrics[name]}"
            )
    diagnostic_paths = {
        "targets": run_dir / "test_targets.npy",
        "predictions": run_dir / "test_predictions.npy",
        "logits": run_dir / "test_logits.npy",
        "spatial": run_dir / "spatial_predictions.npy",
        "spectral": run_dir / "spectral_predictions.npy",
        "gate": run_dir / "gate.npy",
    }
    if all(path.exists() for path in diagnostic_paths.values()):
        targets = np.load(diagnostic_paths["targets"])
        predictions = np.load(diagnostic_paths["predictions"])
        if not np.array_equal(predictions, prediction[valid]):
            raise SystemExit("FAILED: one-dimensional and map predictions differ")
        if not np.array_equal(targets, target[valid]):
            raise SystemExit("FAILED: one-dimensional and map targets differ")
        diagnostics = probabilistic_metrics(targets, np.load(diagnostic_paths["logits"]))
        diagnostics.update(
            routing_diagnostics(
                targets,
                predictions,
                np.load(diagnostic_paths["spatial"]),
                np.load(diagnostic_paths["spectral"]),
                np.load(diagnostic_paths["gate"]),
            )
        )
        for name in ("nll", "brier", "ece", "spatial_branch_oa", "spectral_branch_oa"):
            if not np.isclose(
                diagnostics[name], saved_metrics[name], rtol=0, atol=1e-7, equal_nan=True
            ):
                raise SystemExit(
                    f"FAILED: {name} recomputed as {diagnostics[name]} "
                    f"but metrics.json has {saved_metrics[name]}"
                )
    print(json.dumps(metrics, indent=2))
    print("VERIFIED: prediction -> confusion matrix -> metrics is consistent")


if __name__ == "__main__":
    main()
