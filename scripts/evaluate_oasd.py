from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE = "lassf_mlp_oof_adv_distill_v6_h64"
BASELINE = "lassf_mlp_spatial_only_v6_h64"
UNIFORM = "lassf_mlp_uniform_distill_v6_h64"
REFERENCES = (BASELINE, UNIFORM)


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
    criteria = {
        BASELINE: (0.5, 3, -2.0),
        UNIFORM: (0.0, 3, -2.0),
    }
    for reference_name in REFERENCES:
        difference = _paired(
            candidate, group[group["model"] == reference_name], "oa"
        )
        if len(difference) < 5:
            return {
                "decision": "INCOMPLETE",
                "reason": f"{reference_name} has only {len(difference)} paired seeds",
            }
        mean_threshold, positive_threshold, worst_threshold = criteria[reference_name]
        comparisons[reference_name] = {
            "n_pairs": int(len(difference)),
            "mean_oa_difference_pp": float(100.0 * difference.mean()),
            "positive_pairs": int(np.sum(difference > 0.0)),
            "worst_oa_difference_pp": float(100.0 * difference.min()),
        }
        checks.extend(
            [
                bool(100.0 * difference.mean() >= mean_threshold),
                bool(np.sum(difference > 0.0) >= positive_threshold),
                bool(100.0 * difference.min() >= worst_threshold),
            ]
        )

    baseline = group[group["model"] == BASELINE]
    aa_difference = _paired(candidate, baseline, "aa")
    ece_difference = _paired(candidate, baseline, "ece")
    brier_difference = _paired(candidate, baseline, "brier")
    active_profile_seeds = int(
        np.sum(candidate["distillation_active_class_count"].to_numpy(float) > 0.0)
    )
    checks.extend(
        [
            bool(aa_difference.mean() >= 0.0),
            bool(ece_difference.mean() <= 0.015),
            bool(brier_difference.mean() <= 0.015),
            bool(active_profile_seeds >= 3),
        ]
    )
    return {
        "decision": "DEVELOPMENT_GO" if all(checks) else "DEVELOPMENT_NO_GO",
        "candidate": CANDIDATE,
        "comparisons": comparisons,
        "mean_aa_difference_vs_spatial_pp": float(100.0 * aa_difference.mean()),
        "ece_difference_vs_spatial": float(ece_difference.mean()),
        "brier_difference_vs_spatial": float(brier_difference.mean()),
        "active_profile_seeds": active_profile_seeds,
        "checks_passed": int(sum(checks)),
        "checks_total": len(checks),
        "criteria": {
            "mean_oa_gain_vs_spatial_at_least_pp": 0.5,
            "mean_oa_gain_vs_uniform_at_least_pp": 0.0,
            "positive_pairs_each_reference_at_least": 3,
            "worst_pair_loss_each_reference_no_more_than_pp": 2.0,
            "mean_aa_gain_vs_spatial_non_negative": True,
            "ece_and_brier_tolerance_vs_spatial": 0.015,
            "active_profile_seeds_at_least": 3,
        },
    }


def _markdown(results: dict[str, Any]) -> str:
    lines = [
        "# OASD V6 Development Decision",
        "",
        "This is a pre-registered Pavia development test, not a paper claim.",
        "No test label is used to estimate a distillation weight.",
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
                f"- Seeds with at least one active OOF class: {result['active_profile_seeds']}/5",
                f"- Mean AA gain vs spatial: {result['mean_aa_difference_vs_spatial_pp']:.3f} pp",
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
    parser = argparse.ArgumentParser(description="Apply the OASD-v6 development decision")
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
        "aa",
        "ece",
        "brier",
        "distillation_active_class_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"per_run.csv is missing columns: {missing}")
    results: dict[str, Any] = {}
    for (dataset, protocol), group in frame.groupby(["dataset", "protocol"], sort=True):
        results[f"{dataset}/{protocol}"] = evaluate_group(group)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oasd_development_decision.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "oasd_development_decision.md").write_text(
        _markdown(results), encoding="utf-8"
    )
    print(_markdown(results))


if __name__ == "__main__":
    main()
