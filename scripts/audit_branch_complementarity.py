from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kmfm.artifacts import environment_snapshot, json_dump, sha256_file
from kmfm.complementarity import analyze_seed, summarize_class_rows


PROJECT_DEFAULT = Path("/content/drive/MyDrive/Colab/Unsupervised/KMFM")
VARIANTS = {
    "spatial": "replay_spatial_logit_v4",
    "spectral": "replay_spectral_logit_v4",
    "global": "replay_global_logit_v4",
    "adlf": "replay_adlf_v4",
}


def _seeds(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return values


def _load_run(
    project_root: Path,
    experiment: str,
    dataset: str,
    protocol: str,
    variant: str,
    seed: int,
) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray]:
    run_dir = project_root / "results" / experiment / dataset / protocol / variant / f"seed_{seed}"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    targets_path = run_dir / "test_targets.npy"
    predictions_path = run_dir / "test_predictions.npy"
    missing = [
        str(path)
        for path in (
            status_path,
            metrics_path,
            manifest_path,
            targets_path,
            predictions_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete {variant} seed {seed}; missing: {missing}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "success":
        raise RuntimeError(f"Run is not successful: {run_dir} ({status.get('state')})")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    targets = np.load(targets_path, allow_pickle=False)
    predictions = np.load(predictions_path, allow_pickle=False)
    return run_dir, metrics, targets, predictions


def _validate_seed_inputs(
    loaded: dict[str, tuple[Path, dict[str, Any], np.ndarray, np.ndarray]], seed: int
) -> None:
    reference_name = "spatial"
    _, _, reference_targets, _ = loaded[reference_name]
    reference_manifest = loaded[reference_name][0] / "manifest.json"
    reference_hash = None
    if reference_manifest.is_file():
        manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
        reference_hash = manifest.get("inputs", {}).get("split", {}).get("sha256")
    for name, (run_dir, metrics, targets, predictions) in loaded.items():
        if len(targets) != len(predictions):
            raise ValueError(f"{name} seed {seed} has mismatched target/prediction lengths")
        if not np.array_equal(targets, reference_targets):
            raise ValueError(f"{name} seed {seed} does not use the same test target order")
        if metrics.get("seed") != seed:
            raise ValueError(f"{name} seed {seed} has inconsistent metrics seed")
        manifest_path = run_dir / "manifest.json"
        if reference_hash is not None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split_hash = manifest.get("inputs", {}).get("split", {}).get("sha256")
            if split_hash != reference_hash:
                raise ValueError(f"{name} seed {seed} has a different split hash")


def _markdown(
    *,
    experiment: str,
    dataset: str,
    protocol: str,
    seed_frame: pd.DataFrame,
    class_frame: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Branch Complementarity Audit",
        "",
        "> Development-only diagnostic. Test labels are used to measure oracle ceilings and are not used to select a model or threshold.",
        "",
        f"- Experiment: `{experiment}`",
        f"- Dataset/protocol: `{dataset}/{protocol}`",
        f"- Seeds: `{len(seed_frame)}`",
        "",
        "## Decision",
        "",
        f"- Complementarity: **{decision['complementarity_decision']}**",
        f"- Current global fusion as OA primary: **{decision['global_oa_decision']}**",
        f"- Current ADLF router: **{decision['adlf_decision']}**",
        f"- Prior-risk mismatch signal: **{decision['prior_risk_mismatch_decision']}**",
        f"- Mean oracle gain vs spatial: `{decision['mean_oracle_gain_vs_spatial_pp']:.3f} pp`",
        f"- Mean global OA/AA gain vs spatial: `{decision['mean_global_oa_gain_vs_spatial_pp']:.3f} / {decision['mean_global_aa_gain_vs_spatial_pp']:.3f} pp`",
        f"- Mean class-support/global-gain Spearman: `{decision['mean_global_support_gain_spearman']}`",
        f"- Exclusive correct totals across seed-runs, spectral/spatial: `{decision['exclusive_spectral_correct_total']} / {decision['exclusive_spatial_correct_total']}`",
        "",
        "## Per-Seed Evidence",
        "",
        "| Seed | Spatial OA | Spectral OA | Global OA | ADLF OA | Oracle OA | Oracle gain (pp) | Global AA gain (pp) | Alpha | Rho | Routing AUC | Support-gain rho |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_frame.to_dict(orient="records"):
        def fmt(value: Any, digits: int = 3) -> str:
            return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"

        lines.append(
            f"| {int(row['seed'])} | {100*row['spatial_oa']:.2f}% | {100*row['spectral_oa']:.2f}% | "
            f"{100*row.get('global_oa', float('nan')):.2f}% | {100*row.get('adlf_oa', float('nan')):.2f}% | "
            f"{100*row['oracle_oa']:.2f}% | {row['oracle_gain_vs_spatial_pp']:.2f} | "
            f"{row.get('global_aa_gain_vs_spatial_pp', float('nan')):.2f} | "
            f"{fmt(row.get('selected_alpha'))} | {fmt(row.get('selected_radius'))} | "
            f"{fmt(row.get('routing_auc'))} | {fmt(row.get('global_support_gain_spearman'))} |"
        )
    lines.extend(
        [
            "",
            "## Class-Level Evidence",
            "",
            "`oracle_gain_vs_spatial_pp_mean` is the per-class oracle upper bound, not an achieved method result.",
            "",
            "| Class | Mean support | Spatial acc. | Spectral acc. | Oracle acc. | Oracle gain (pp) | Spectral-only correct | Spatial-only correct |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_frame.to_dict(orient="records"):
        lines.append(
            f"| {int(row['class_index'])} | {row['support_mean']:.1f} | "
            f"{100*row['spatial_accuracy_mean']:.2f}% | {100*row['spectral_accuracy_mean']:.2f}% | "
            f"{100*row['oracle_accuracy_mean']:.2f}% | {row['oracle_gain_vs_spatial_pp_mean']:.2f} | "
            f"{row['exclusive_spectral_correct_mean']:.1f} | {row['exclusive_spatial_correct_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The oracle columns quantify whether a selector is worth researching; they do not establish a deployable method.",
            "- The next confirmatory method must be spatial-anchored, abstain by default, and learn correction benefit from training-region out-of-fold predictions.",
            "- Pavia remains a development diagnostic after this audit; no threshold is to be selected from these test labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit branch complementarity without selecting a model")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--experiment", default="pavia_adlf_replay_v4")
    parser.add_argument("--dataset", default="pavia_university")
    parser.add_argument("--protocol", default="spatial_block")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0,1,2,3,4"))
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir or project_root / "reports" / args.experiment / "complementarity_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for seed in args.seeds:
        loaded: dict[str, tuple[Path, dict[str, Any], np.ndarray, np.ndarray]] = {}
        for name, variant in VARIANTS.items():
            loaded[name] = _load_run(
                project_root, args.experiment, args.dataset, args.protocol, variant, seed
            )
            run_dir = loaded[name][0]
            sources[f"{name}/seed_{seed}"] = {
                "run_dir": str(run_dir),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "targets_sha256": sha256_file(run_dir / "test_targets.npy"),
                "predictions_sha256": sha256_file(run_dir / "test_predictions.npy"),
            }
        _validate_seed_inputs(loaded, seed)
        _, adlf_metrics, _, adlf_predictions = loaded["adlf"]
        _, _, targets, spatial_predictions = loaded["spatial"]
        _, _, _, spectral_predictions = loaded["spectral"]
        _, _, _, global_predictions = loaded["global"]
        seed_row, rows = analyze_seed(
            targets,
            spatial_predictions,
            spectral_predictions,
            global_predictions=global_predictions,
            adlf_predictions=adlf_predictions,
            selected_alpha=adlf_metrics.get("selected_alpha"),
            selected_radius=adlf_metrics.get("selected_radius"),
        )
        seed_row["seed"] = seed
        seed_row["dataset"] = args.dataset
        seed_row["protocol"] = args.protocol
        seed_row["routing_auc"] = adlf_metrics.get("routing_auc")
        for row in rows:
            row["seed"] = seed
            row["dataset"] = args.dataset
            row["protocol"] = args.protocol
        seed_rows.append(seed_row)
        class_rows.extend(rows)

    seed_frame = pd.DataFrame(seed_rows).sort_values("seed")
    class_summary = pd.DataFrame(summarize_class_rows(class_rows)).sort_values("class_index")
    mean_oracle_gain = float(seed_frame["oracle_gain_vs_spatial_pp"].mean())
    positive_oracle_seeds = int((seed_frame["oracle_gain_vs_spatial_pp"] > 0).sum())
    global_oa_gain = float(seed_frame["global_oa_gain_vs_spatial_pp"].mean())
    global_aa_gain = float(seed_frame["global_aa_gain_vs_spatial_pp"].mean())
    adlf_global_gain = float(
        (seed_frame["adlf_oa"] - seed_frame["global_oa"]).mean() * 100.0
    )
    adlf_spatial_gain = float(seed_frame["adlf_oa_gain_vs_spatial_pp"].mean())
    prior_mismatch_count = int(seed_frame["global_aa_positive_oa_negative"].sum())
    support_gain_values = seed_frame["global_support_gain_spearman"].dropna().to_numpy(
        dtype=float
    )
    mean_support_gain_spearman = (
        float(support_gain_values.mean()) if len(support_gain_values) else None
    )
    required_positive_oracle_seeds = max(1, int(np.ceil(0.8 * len(seed_frame))))
    decision = {
        "complementarity_decision": (
            "CONTINUE_SELECTOR_DEVELOPMENT"
            if mean_oracle_gain >= 3.0
            and positive_oracle_seeds >= required_positive_oracle_seeds
            else "STOP_DUAL_BRANCH"
        ),
        "global_oa_decision": "UNSUITABLE_AS_PRIMARY" if global_oa_gain <= 0.0 else "RETAIN_AS_BASELINE",
        "adlf_decision": (
            "NO_GO_AS_CURRENT_ROUTER"
            if adlf_global_gain <= 0.0 or adlf_spatial_gain <= 0.0
            else "REQUIRES_FRESH_CONFIRMATION"
        ),
        "prior_risk_mismatch_decision": (
            "SUPPORTED_FOR_CLASSWISE_AUDIT" if prior_mismatch_count >= 3 else "NOT_ESTABLISHED"
        ),
        "mean_oracle_gain_vs_spatial_pp": mean_oracle_gain,
        "positive_oracle_seed_count": positive_oracle_seeds,
        "mean_global_oa_gain_vs_spatial_pp": global_oa_gain,
        "mean_global_aa_gain_vs_spatial_pp": global_aa_gain,
        "mean_adlf_oa_gain_vs_global_pp": adlf_global_gain,
        "mean_adlf_oa_gain_vs_spatial_pp": adlf_spatial_gain,
        "mean_global_support_gain_spearman": mean_support_gain_spearman,
        "exclusive_spectral_correct_total": int(
            seed_frame["exclusive_spectral_correct"].sum()
        ),
        "exclusive_spatial_correct_total": int(
            seed_frame["exclusive_spatial_correct"].sum()
        ),
        "prior_risk_mismatch_seed_count": prior_mismatch_count,
        "criteria": {
            "continue_mean_oracle_gain_at_least_pp": 3.0,
            "continue_positive_oracle_seeds_at_least": required_positive_oracle_seeds,
            "prior_mismatch_seeds_for_signal": 3,
        },
    }
    seed_frame.to_csv(output_dir / "complementarity_per_seed.csv", index=False)
    pd.DataFrame(class_rows).sort_values(["seed", "class_index"]).to_csv(
        output_dir / "complementarity_per_class_seed.csv", index=False
    )
    class_summary.to_csv(output_dir / "complementarity_per_class_summary.csv", index=False)
    payload = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "experiment": args.experiment,
        "seeds": args.seeds,
        "decision": decision,
        "sources": sources,
        "environment": environment_snapshot(),
    }
    json_dump(output_dir / "complementarity_audit.json", payload)
    (output_dir / "complementarity_audit.md").write_text(
        _markdown(
            experiment=args.experiment,
            dataset=args.dataset,
            protocol=args.protocol,
            seed_frame=seed_frame,
            class_frame=class_summary,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print((output_dir / "complementarity_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
