from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = ("oa", "aa", "kappa")
PROBABILISTIC_METRICS = ("nll", "brier", "ece")
DIAGNOSTIC_FIELDS = (
    "spatial_branch_oa",
    "spectral_branch_oa",
    "spatial_branch_nll",
    "spectral_branch_nll",
    "oracle_oa",
    "oracle_gap",
    "oracle_gap_recovery",
    "discordant_count",
    "discordant_fraction",
    "routing_auc",
    "contribution_ratio_mean",
    "contribution_ratio_std",
    "gate_entropy_gap_spearman",
    "spatial_temperature",
    "spectral_temperature",
    "selected_alpha",
    "selected_radius",
    "validation_nll",
    "prediction_disagreement_count",
    "prediction_disagreement_fraction",
    "vs_global_improved_count",
    "vs_global_harmed_count",
    "vs_global_net_corrected",
    "source_parameter_count",
    "selector_parameter_count",
    "selector_policy_seconds",
    "test_switch_count",
    "test_switch_fraction",
    "vs_spatial_improved_count",
    "vs_spatial_harmed_count",
    "vs_spatial_net_corrected",
    "oracle_recovery_vs_spatial",
    "selector_score_mean",
    "selector_score_std",
    "selected_correction_threshold",
    "selected_correction_coverage",
    "validation_oof_disagreement_count",
    "validation_oof_selected_count",
    "validation_oof_improved_count",
    "validation_oof_harmed_count",
    "validation_oof_neutral_count",
    "validation_oof_net_corrected",
    "validation_oof_wilson_lower",
)
PAIRED_METRICS = PRIMARY_METRICS + PROBABILISTIC_METRICS


def _load_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "success":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        row = {
            "run_dir": str(run_dir),
            "dataset": metrics["dataset"],
            "protocol": metrics["protocol"],
            "model": metrics["model"],
            "seed": int(metrics["seed"]),
            "split_sha256": manifest.get("inputs", {}).get("split", {}).get("sha256"),
            "parameter_count": metrics.get("parameter_count"),
            "training_seconds": metrics.get("training_seconds"),
            "test_inference_seconds": metrics.get("test_inference_seconds"),
            "gate_mean": metrics.get("gate_mean"),
            "gate_std": metrics.get("gate_std"),
        }
        row.update({metric: metrics[metric] for metric in PRIMARY_METRICS})
        for field in PROBABILISTIC_METRICS + DIAGNOSTIC_FIELDS:
            row[field] = metrics.get(field)
        rows.append(row)
    if not rows:
        raise ValueError(f"No successful metrics.json files found under {root}")
    frame = pd.DataFrame(rows)
    duplicate = frame.duplicated(["dataset", "protocol", "model", "seed"], keep=False)
    if duplicate.any():
        conflicts = frame.loc[duplicate, ["dataset", "protocol", "model", "seed", "run_dir"]]
        raise ValueError(
            "Duplicate successful runs for the same dataset/protocol/model/seed. "
            "Use a clean experiment root instead of selecting a preferred rerun:\n"
            + conflicts.to_string(index=False)
        )
    return frame.sort_values(["dataset", "protocol", "model", "seed"])


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["dataset", "protocol", "model"], sort=True):
        record: dict[str, Any] = {
            "dataset": keys[0],
            "protocol": keys[1],
            "model": keys[2],
            "n": len(group),
        }
        for metric in PRIMARY_METRICS + PROBABILISTIC_METRICS + DIAGNOSTIC_FIELDS:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            half = (
                float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_ci95_low"] = mean - half
            record[f"{metric}_ci95_high"] = mean + half
            if metric in PRIMARY_METRICS:
                record[f"{metric}_mean_percent"] = 100.0 * mean
                record[f"{metric}_sd_percent"] = 100.0 * sd
                record[f"{metric}_ci95_low_percent"] = 100.0 * (mean - half)
                record[f"{metric}_ci95_high_percent"] = 100.0 * (mean + half)
        record["oa_min"] = float(group["oa"].min())
        record["oa_max"] = float(group["oa"].max())
        record["parameter_count_mean"] = float(group["parameter_count"].mean())
        record["training_seconds_mean"] = float(group["training_seconds"].mean())
        record["test_inference_seconds_mean"] = float(group["test_inference_seconds"].mean())
        records.append(record)
    return pd.DataFrame(records)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_tests(frame: pd.DataFrame, reference_models: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset, protocol), subset in frame.groupby(["dataset", "protocol"], sort=True):
        for reference_model in reference_models:
            reference = subset[subset["model"] == reference_model].set_index("seed")
            if reference.empty:
                continue
            for model in sorted(set(subset["model"]) - {reference_model}):
                candidate = subset[subset["model"] == model].set_index("seed")
                common = sorted(set(reference.index) & set(candidate.index))
                if len(common) < 2:
                    continue
                for seed in common:
                    reference_hash = reference.loc[seed, "split_sha256"]
                    candidate_hash = candidate.loc[seed, "split_sha256"]
                    if (
                        pd.notna(reference_hash)
                        and pd.notna(candidate_hash)
                        and reference_hash != candidate_hash
                    ):
                        raise ValueError(
                            f"Paired comparison uses different split hashes for "
                            f"{dataset}/{protocol}/seed {seed}: {reference_model}={reference_hash}, "
                            f"{model}={candidate_hash}"
                        )
                for metric in PAIRED_METRICS:
                    paired = pd.concat(
                        [candidate.loc[common, metric], reference.loc[common, metric]],
                        axis=1,
                    ).dropna()
                    if len(paired) < 2:
                        continue
                    x = paired.iloc[:, 0].to_numpy(dtype=float)
                    y = paired.iloc[:, 1].to_numpy(dtype=float)
                    difference = x - y
                    t_p = float(stats.ttest_rel(x, y).pvalue)
                    try:
                        w_p = (
                            float(stats.wilcoxon(difference).pvalue)
                            if np.any(difference != 0)
                            else 1.0
                        )
                    except ValueError:
                        w_p = float("nan")
                    difference_sd = difference.std(ddof=1)
                    dz = float(difference.mean() / difference_sd) if difference_sd > 0 else 0.0
                    records.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "model": model,
                            "reference": reference_model,
                            "metric": metric,
                            "n_pairs": len(paired),
                            "mean_difference": float(difference.mean()),
                            "mean_difference_percentage_points": (
                                float(100.0 * difference.mean())
                                if metric in PRIMARY_METRICS
                                else None
                            ),
                            "positive_pairs": int(np.sum(difference > 0)),
                            "negative_pairs": int(np.sum(difference < 0)),
                            "paired_t_p": t_p,
                            "wilcoxon_p": w_p,
                            "cohen_dz": dz,
                        }
                    )
    if not records:
        return pd.DataFrame()
    tests = pd.DataFrame(records)
    tests["paired_t_p_holm"] = _holm_adjust(tests["paired_t_p"].fillna(1.0).tolist())
    tests["wilcoxon_p_holm"] = _holm_adjust(tests["wilcoxon_p"].fillna(1.0).tolist())
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate immutable KMFM rebuild runs")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-model")
    args = parser.parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    summary = _summary(runs)
    runs.to_csv(output / "per_run.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if args.reference_model:
        reference_models = [
            value.strip() for value in args.reference_model.split(",") if value.strip()
        ]
        tests = _paired_tests(runs, reference_models)
        tests.to_csv(output / "paired_tests.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
