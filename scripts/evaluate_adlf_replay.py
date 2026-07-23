from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE = "replay_adlf_v4"
REFERENCES = ("replay_spatial_logit_v4", "replay_global_logit_v4")


def _paired(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> np.ndarray:
    left = candidate.set_index("seed")[[metric]].rename(columns={metric: "candidate"})
    right = reference.set_index("seed")[[metric]].rename(columns={metric: "reference"})
    joined = left.join(right, how="inner").dropna()
    return joined["candidate"].to_numpy(dtype=float) - joined["reference"].to_numpy(
        dtype=float
    )


def evaluate_group(group: pd.DataFrame) -> dict[str, Any]:
    models = set(group["model"])
    missing = [model for model in (CANDIDATE, *REFERENCES) if model not in models]
    if missing:
        return {"decision": "INCOMPLETE", "missing_models": missing}

    candidate = group[group["model"] == CANDIDATE]
    comparisons: dict[str, Any] = {}
    checks: list[bool] = []
    for reference_name in REFERENCES:
        reference = group[group["model"] == reference_name]
        difference = _paired(candidate, reference, "oa")
        if len(difference) < 5:
            return {
                "decision": "INCOMPLETE",
                "reason": f"{reference_name} has only {len(difference)} paired seeds",
            }
        comparisons[reference_name] = {
            "n_pairs": int(len(difference)),
            "mean_oa_difference": float(difference.mean()),
            "mean_oa_difference_pp": float(100.0 * difference.mean()),
            "positive_pairs": int(np.sum(difference > 0.0)),
            "worst_oa_difference": float(difference.min()),
            "worst_oa_difference_pp": float(100.0 * difference.min()),
        }
        checks.extend(
            [
                bool(difference.mean() >= 0.01),
                bool(np.sum(difference > 0.0) >= 4),
                bool(difference.min() >= -0.015),
            ]
        )

    global_reference = group[group["model"] == "replay_global_logit_v4"]
    ece_difference = _paired(candidate, global_reference, "ece")
    brier_difference = _paired(candidate, global_reference, "brier")
    active_routing_seeds = int(np.sum(candidate["selected_radius"].to_numpy(float) > 0.0))
    checks.extend(
        [
            bool(ece_difference.mean() <= 0.01),
            bool(brier_difference.mean() <= 0.01),
            bool(active_routing_seeds >= 3),
        ]
    )

    routing_auc = candidate["routing_auc"].dropna().to_numpy(dtype=float)
    return {
        "decision": "GO" if all(checks) else "NO_GO",
        "candidate": CANDIDATE,
        "comparisons": comparisons,
        "ece_difference_vs_global": float(ece_difference.mean()),
        "brier_difference_vs_global": float(brier_difference.mean()),
        "active_routing_seeds": active_routing_seeds,
        "routing_auc_mean": float(routing_auc.mean()) if len(routing_auc) else None,
        "checks_passed": int(sum(checks)),
        "checks_total": int(len(checks)),
        "criteria": {
            "mean_oa_gain_each_reference_at_least_pp": 1.0,
            "positive_pairs_each_reference_at_least": 4,
            "worst_pair_loss_each_reference_no_more_than_pp": 1.5,
            "ece_and_brier_tolerance_vs_global": 0.01,
            "active_routing_seeds_at_least": 3,
        },
    }


def _markdown(results: dict[str, Any]) -> str:
    lines = [
        "# ADLF Replay Decision",
        "",
        "This is a pre-registered checkpoint-replay feasibility decision, not a paper claim.",
        "",
    ]
    for key, result in results.items():
        lines.extend([f"## {key}", "", f"**Decision: {result['decision']}**", ""])
        if result["decision"] == "INCOMPLETE":
            lines.extend([f"Reason: {result.get('reason', result.get('missing_models'))}", ""])
            continue
        lines.extend(
            [
                f"- Checks passed: {result['checks_passed']}/{result['checks_total']}",
                f"- Active routing seeds: {result['active_routing_seeds']}/5",
                f"- Routing AUC mean (diagnostic only): {result['routing_auc_mean']}",
                f"- ECE difference vs global: {result['ece_difference_vs_global']:.6f}",
                f"- Brier difference vs global: {result['brier_difference_vs_global']:.6f}",
                "",
                "| Reference | Mean OA gain (pp) | Positive pairs | Worst pair (pp) |",
                "|---|---:|---:|---:|",
            ]
        )
        for reference, comparison in result["comparisons"].items():
            lines.append(
                f"| {reference} | {comparison['mean_oa_difference_pp']:.3f} | "
                f"{comparison['positive_pairs']}/{comparison['n_pairs']} | "
                f"{comparison['worst_oa_difference_pp']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-registered ADLF replay decision")
    parser.add_argument("--per-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.per_run)
    required = {"dataset", "protocol", "model", "seed", "oa", "ece", "brier", "routing_auc", "selected_radius"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"per_run.csv is missing columns: {missing}")

    results: dict[str, Any] = {}
    for (dataset, protocol), group in frame.groupby(["dataset", "protocol"], sort=True):
        results[f"{dataset}/{protocol}"] = evaluate_group(group)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adlf_replay_decision.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "adlf_replay_decision.md").write_text(
        _markdown(results), encoding="utf-8"
    )
    print(_markdown(results))


if __name__ == "__main__":
    main()
