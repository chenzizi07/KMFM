from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE = "replay_ssrc_class_v5"
REFERENCES = (
    "replay_spatial_logit_v5",
    "replay_global_logit_v5",
    "replay_ssrc_score_v5",
)


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
    checks: list[bool] = []
    comparisons: dict[str, Any] = {}
    criteria = {
        "replay_spatial_logit_v5": (1.0, 4, -1.0),
        "replay_global_logit_v5": (1.0, 4, -1.0),
        "replay_ssrc_score_v5": (0.5, 3, -1.0),
    }
    for reference_name in REFERENCES:
        reference = group[group["model"] == reference_name]
        difference = _paired(candidate, reference, "oa")
        if len(difference) < 5:
            return {
                "decision": "INCOMPLETE",
                "reason": f"{reference_name} has only {len(difference)} paired seeds",
            }
        mean_threshold_pp, positive_threshold, worst_threshold_pp = criteria[reference_name]
        comparisons[reference_name] = {
            "n_pairs": int(len(difference)),
            "mean_oa_difference_pp": float(100.0 * difference.mean()),
            "positive_pairs": int(np.sum(difference > 0.0)),
            "worst_oa_difference_pp": float(100.0 * difference.min()),
        }
        checks.extend(
            [
                bool(100.0 * difference.mean() >= mean_threshold_pp),
                bool(np.sum(difference > 0.0) >= positive_threshold),
                bool(100.0 * difference.min() >= worst_threshold_pp),
            ]
        )

    active_seeds = int(np.sum(candidate["test_switch_count"].to_numpy(float) > 0.0))
    recoveries = candidate["oracle_recovery_vs_spatial"].dropna().to_numpy(float)
    mean_recovery = float(recoveries.mean()) if len(recoveries) else None
    spatial = group[group["model"] == "replay_spatial_logit_v5"]
    ece_difference = _paired(candidate, spatial, "ece")
    brier_difference = _paired(candidate, spatial, "brier")
    checks.extend(
        [
            bool(active_seeds >= 3),
            bool(mean_recovery is not None and mean_recovery >= 0.15),
            bool(ece_difference.mean() <= 0.01),
            bool(brier_difference.mean() <= 0.01),
        ]
    )
    return {
        "decision": "DEVELOPMENT_GO" if all(checks) else "DEVELOPMENT_NO_GO",
        "candidate": CANDIDATE,
        "comparisons": comparisons,
        "active_selector_seeds": active_seeds,
        "mean_oracle_recovery_vs_spatial": mean_recovery,
        "ece_difference_vs_spatial": float(ece_difference.mean()),
        "brier_difference_vs_spatial": float(brier_difference.mean()),
        "checks_passed": int(sum(checks)),
        "checks_total": int(len(checks)),
        "criteria": {
            "mean_oa_gain_vs_spatial_and_global_at_least_pp": 1.0,
            "positive_pairs_vs_spatial_and_global_at_least": 4,
            "mean_oa_gain_vs_score_only_at_least_pp": 0.5,
            "positive_pairs_vs_score_only_at_least": 3,
            "worst_pair_loss_each_reference_no_more_than_pp": 1.0,
            "active_selector_seeds_at_least": 3,
            "mean_oracle_recovery_at_least": 0.15,
            "ece_and_brier_tolerance_vs_spatial": 0.01,
        },
    }


def _markdown(results: dict[str, Any]) -> str:
    lines = [
        "# SSRC V5 Development Decision",
        "",
        "This is a development-only replay on a previously inspected dataset, not a paper claim.",
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
                f"- Active selector seeds: {result['active_selector_seeds']}/5",
                f"- Mean oracle recovery vs spatial: {result['mean_oracle_recovery_vs_spatial']}",
                f"- ECE difference vs spatial: {result['ece_difference_vs_spatial']:.6f}",
                f"- Brier difference vs spatial: {result['brier_difference_vs_spatial']:.6f}",
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
    parser = argparse.ArgumentParser(description="Apply the SSRC-v5 development decision")
    parser.add_argument("--per-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.per_run)
    required = {
        "dataset",
        "protocol",
        "model",
        "seed",
        "oa",
        "ece",
        "brier",
        "test_switch_count",
        "oracle_recovery_vs_spatial",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"per_run.csv is missing columns: {missing}")
    results: dict[str, Any] = {}
    for (dataset, protocol), group in frame.groupby(["dataset", "protocol"], sort=True):
        results[f"{dataset}/{protocol}"] = evaluate_group(group)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ssrc_development_decision.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "ssrc_development_decision.md").write_text(
        _markdown(results), encoding="utf-8"
    )
    print(_markdown(results))


if __name__ == "__main__":
    main()
