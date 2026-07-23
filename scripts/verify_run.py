from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kmfm.metrics import classification_metrics


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
    print(json.dumps(metrics, indent=2))
    print("VERIFIED: prediction -> confusion matrix -> metrics is consistent")


if __name__ == "__main__":
    main()
