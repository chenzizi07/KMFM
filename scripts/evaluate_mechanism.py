from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATE = "lassf_mlp_entropy_softmax_v3_h64"
REFERENCES = (
    "lassf_mlp_spatial_only_v3_h64",
    "lassf_mlp_gate_norm_v3_h64",
)


def _paired(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> np.ndarray:
    merged = candidate[["seed", metric]].merge(
        reference[["seed", metric]], on="seed", suffixes=("_candidate", "_reference")
    )
    return (
        merged[f"{metric}_candidate"].to_numpy(dtype=float)
        - merged[f"{metric}_reference"].to_numpy(dtype=float)
    )


def evaluate_group(group: pd.DataFrame) -> dict[str, Any]:
    candidate = group[group["model"] == CANDIDATE]
    missing = [model for model in (CANDIDATE, *REFERENCES) if model not in set(group["model"])]
    if missing:
        return {"decision": "INCOMPLETE", "missing_models": missing}

    comparisons: dict[str, Any] = {}
    checks: list[bool] = []
    for reference_name in REFERENCES:
        reference = group[group["model"] == reference_name]
        oa_difference = _paired(candidate, reference, "oa")
        if len(oa_difference) < 5:
            return {
                "decision": "INCOMPLETE",
                "reason": f"{reference_name} has only {len(oa_difference)} paired splits",
            }
        comparisons[reference_name] = {
            "n_pairs": int(len(oa_difference)),
            "mean_oa_difference": float(oa_difference.mean()),
            "mean_oa_difference_pp": float(100.0 * oa_difference.mean()),
            "positive_pairs": int(np.sum(oa_difference > 0)),
            "worst_oa_difference": float(oa_difference.min()),
            "worst_oa_difference_pp": float(100.0 * oa_difference.min()),
        }
        checks.extend(
            [
                bool(oa_difference.mean() >= 0.01),
                bool(np.sum(oa_difference > 0) >= 4),
                bool(oa_difference.min() >= -0.02),
            ]
        )

    routing_auc = candidate["routing_auc"].dropna().to_numpy(dtype=float)
    routing_auc_mean = float(routing_auc.mean()) if len(routing_auc) else None
    checks.append(bool(routing_auc_mean is not None and routing_auc_mean >= 0.60))

    gate_reference = group[group["model"] == "lassf_mlp_gate_norm_v3_h64"]
    ece_difference = _paired(candidate, gate_reference, "ece")
    brier_difference = _paired(candidate, gate_reference, "brier")
    checks.extend(
        [
            bool(ece_difference.mean() <= 0.01),
            bool(brier_difference.mean() <= 0.01),
        ]
    )

    return {
        "decision": "GO" if all(checks) else "NO_GO",
        "candidate": CANDIDATE,
        "comparisons": comparisons,
        "routing_auc_mean": routing_auc_mean,
        "ece_difference_vs_plain_gate": float(ece_difference.mean()),
        "brier_difference_vs_plain_gate": float(brier_difference.mean()),
        "criteria": {
            "mean_oa_gain_each_reference_at_least_pp": 1.0,
            "positive_pairs_each_reference_at_least": 4,
            "worst_pair_loss_each_reference_no_more_than_pp": 2.0,
            "routing_auc_at_least": 0.60,
            "ece_and_brier_tolerance_vs_plain_gate": 0.01,
        },
        "checks_passed": int(sum(checks)),
        "checks_total": int(len(checks)),
    }


def _markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Calibrated V3 Mechanism Decision",
        "",
        "This is a pre-registered pilot decision, not a statistical significance claim.",
        "",
    ]
    for key, result in results.items():
        lines.extend([f"## {key}", "", f"**Decision: {result['decision']}**", ""])
        if result["decision"] == "INCOMPLETE":
            lines.append(f"Reason: {result.get('reason', result.get('missing_models'))}")
            lines.append("")
            continue
        lines.extend(
            [
                f"- Routing AUC mean: {result['routing_auc_mean']}",
                f"- ECE difference vs plain gate: {result['ece_difference_vs_plain_gate']:.6f}",
                f"- Brier difference vs plain gate: {result['brier_difference_vs_plain_gate']:.6f}",
                f"- Checks passed: {result['checks_passed']}/{result['checks_total']}",
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
    parser = argparse.ArgumentParser(description="Apply the pre-registered calibrated-v3 decision")
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
        "routing_auc",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"per_run.csv is missing columns: {missing_columns}")

    results: dict[str, Any] = {}
    for (dataset, protocol), group in frame.groupby(["dataset", "protocol"], sort=True):
        results[f"{dataset}/{protocol}"] = evaluate_group(group)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mechanism_decision.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "mechanism_decision.md").write_text(_markdown(results), encoding="utf-8")
    print(_markdown(results))


if __name__ == "__main__":
    main()
