from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from kmfm.engine import _spearman
from kmfm.metrics import classification_metrics, probabilistic_metrics, routing_diagnostics
from kmfm.selective_correction import SSRCPolicy, apply_ssrc_policy, fit_ssrc_policy
from scripts.replay_logit_fusion import (
    PROJECT_DEFAULT,
    _archive_incomplete,
    _load_source_outputs,
    _save_variant,
    _seeds,
    _status,
    probabilistic_metrics_per_sample_entropy,
)


VARIANTS = (
    "replay_spatial_logit_v5",
    "replay_spectral_logit_v5",
    "replay_global_logit_v5",
    "replay_ssrc_score_v5",
    "replay_ssrc_class_v5",
)


def _rule_for_variant(variant: str, policy: SSRCPolicy):
    if variant == "replay_ssrc_score_v5":
        return policy.score_only
    if variant == "replay_ssrc_class_v5":
        return policy.class_aware
    return None


def _variant_alpha(variant: str, policy: SSRCPolicy) -> float:
    return {
        "replay_spatial_logit_v5": 1.0,
        "replay_spectral_logit_v5": 0.0,
        "replay_global_logit_v5": policy.global_alpha,
        "replay_ssrc_score_v5": 1.0,
        "replay_ssrc_class_v5": 1.0,
    }[variant]


def _validation_nll(variant: str, policy: SSRCPolicy) -> float | None:
    key = {
        "replay_spatial_logit_v5": "spatial",
        "replay_spectral_logit_v5": "spectral",
        "replay_global_logit_v5": "global",
    }.get(variant)
    return policy.validation_nll[key] if key is not None else None


def _selector_parameter_count(variant: str, policy: SSRCPolicy) -> int:
    rule = _rule_for_variant(variant, policy)
    return len(rule.model.coefficients) + 1 if rule is not None else 0


def _variant_metrics(
    *,
    variant: str,
    seed: int,
    dataset: str,
    protocol: str,
    logits: np.ndarray,
    weights: np.ndarray,
    selector_scores: np.ndarray | None,
    targets: np.ndarray,
    spatial_logits: np.ndarray,
    spectral_logits: np.ndarray,
    global_predictions: np.ndarray,
    policy: SSRCPolicy,
    source_inference_seconds: float,
    selector_policy_seconds: float,
    source_parameter_count: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predictions = np.asarray(logits).argmax(axis=1)
    spatial_predictions = spatial_logits.argmax(axis=1)
    spectral_predictions = spectral_logits.argmax(axis=1)
    metrics, confusion = classification_metrics(targets, predictions, logits.shape[1])
    metrics.update(probabilistic_metrics(targets, logits))
    routing = routing_diagnostics(
        targets, predictions, spatial_predictions, spectral_predictions, weights
    )
    metrics.update(routing)
    spatial_probability = probabilistic_metrics(targets, spatial_logits)
    spectral_probability = probabilistic_metrics(targets, spectral_logits)
    metrics.update(
        {f"spatial_branch_{name}": value for name, value in spatial_probability.items()}
    )
    metrics.update(
        {f"spectral_branch_{name}": value for name, value in spectral_probability.items()}
    )

    spatial_correct = spatial_predictions == targets
    global_correct = global_predictions == targets
    candidate_correct = predictions == targets
    disagreement = spatial_predictions != spectral_predictions
    rule = _rule_for_variant(variant, policy)
    selected = (
        np.asarray(weights) < 0.5
        if rule is not None
        else np.zeros(len(weights), dtype=bool)
    )
    selector_parameters = _selector_parameter_count(variant, policy)
    oracle_gain_vs_spatial = routing["oracle_oa"] - routing["spatial_branch_oa"]
    oracle_recovery_vs_spatial = (
        (metrics["oa"] - routing["spatial_branch_oa"]) / oracle_gain_vs_spatial
        if oracle_gain_vs_spatial > 1e-12
        else None
    )
    metrics.update(
        {
            "dataset": dataset,
            "protocol": protocol,
            "model": variant,
            "seed": seed,
            "source_parameter_count": source_parameter_count,
            "selector_parameter_count": selector_parameters,
            "parameter_count": source_parameter_count + selector_parameters,
            "training_seconds": 0.0,
            "test_inference_seconds": source_inference_seconds,
            "selector_policy_seconds": selector_policy_seconds if rule is not None else 0.0,
            "gate_mean": float(np.mean(weights)),
            "gate_std": float(np.std(weights)),
            "contribution_ratio_mean": float(np.mean(weights)),
            "contribution_ratio_std": float(np.std(weights)),
            "gate_entropy_gap_spearman": _spearman(
                weights,
                probabilistic_metrics_per_sample_entropy(spectral_logits)
                - probabilistic_metrics_per_sample_entropy(spatial_logits),
            ),
            "spatial_temperature": policy.spatial_temperature,
            "spectral_temperature": policy.spectral_temperature,
            "selected_alpha": _variant_alpha(variant, policy),
            "selected_radius": 0.0,
            "validation_nll": _validation_nll(variant, policy),
            "prediction_disagreement_count": int(disagreement.sum()),
            "prediction_disagreement_fraction": float(disagreement.mean()),
            "test_switch_count": int(selected.sum()),
            "test_switch_fraction": float(selected.mean()),
            "vs_spatial_improved_count": int((~spatial_correct & candidate_correct).sum()),
            "vs_spatial_harmed_count": int((spatial_correct & ~candidate_correct).sum()),
            "vs_spatial_net_corrected": int(
                (~spatial_correct & candidate_correct).sum()
                - (spatial_correct & ~candidate_correct).sum()
            ),
            "vs_global_improved_count": int((~global_correct & candidate_correct).sum()),
            "vs_global_harmed_count": int((global_correct & ~candidate_correct).sum()),
            "vs_global_net_corrected": int(
                (~global_correct & candidate_correct).sum()
                - (global_correct & ~candidate_correct).sum()
            ),
            "oracle_recovery_vs_spatial": oracle_recovery_vs_spatial,
            "selector_score_mean": (
                float(np.mean(selector_scores)) if selector_scores is not None else None
            ),
            "selector_score_std": (
                float(np.std(selector_scores)) if selector_scores is not None else None
            ),
            "selected_correction_threshold": rule.threshold if rule is not None else None,
            "selected_correction_coverage": (
                rule.target_coverage if rule is not None else 0.0
            ),
            "validation_oof_disagreement_count": (
                rule.oof_disagreement_count if rule is not None else None
            ),
            "validation_oof_selected_count": (
                rule.oof_selected_count if rule is not None else None
            ),
            "validation_oof_improved_count": (
                rule.oof_improved_count if rule is not None else None
            ),
            "validation_oof_harmed_count": (
                rule.oof_harmed_count if rule is not None else None
            ),
            "validation_oof_neutral_count": (
                rule.oof_neutral_count if rule is not None else None
            ),
            "validation_oof_net_corrected": (
                rule.oof_net_corrected if rule is not None else None
            ),
            "validation_oof_wilson_lower": (
                rule.oof_wilson_lower if rule is not None else None
            ),
        }
    )
    return metrics, confusion, predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development replay for spatial-anchored selective spectral correction"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--source-experiment", default="pavia_calibrated_v3")
    parser.add_argument("--output-experiment", default="pavia_ssrc_dev_v5")
    parser.add_argument("--dataset", default="pavia_university")
    parser.add_argument("--protocol", default="spatial_block")
    parser.add_argument("--source-model", default="lassf_mlp_concat_norm_v3_h64")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0,1,2,3,4"))
    parser.add_argument("--recover-incomplete", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_root = project_root / "results"
    output_root = results_root / args.output_experiment
    for seed in args.seeds:
        run_dirs = {
            variant: output_root / args.dataset / args.protocol / variant / f"seed_{seed}"
            for variant in VARIANTS
        }
        pending: list[str] = []
        for variant, run_dir in run_dirs.items():
            status = _status(run_dir)
            if status == "success":
                print(f"SKIP successful: {run_dir}")
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                if not args.recover_incomplete:
                    raise RuntimeError(
                        f"Incomplete immutable SSRC replay exists ({status}): {run_dir}. "
                        "Inspect it or pass --recover-incomplete."
                    )
                archive = _archive_incomplete(
                    run_dir, results_root, args.output_experiment
                )
                print(f"ARCHIVE incomplete ({status}): {run_dir} -> {archive}")
            pending.append(variant)
        if not pending:
            continue

        source_run = (
            results_root
            / args.source_experiment
            / args.dataset
            / args.protocol
            / args.source_model
            / f"seed_{seed}"
        )
        print(f"SSRC DEV {args.dataset} {args.protocol} {args.source_model} seed={seed}")
        source_config, val_result, test_result, inference_seconds, parameter_count = (
            _load_source_outputs(source_run)
        )
        policy = fit_ssrc_policy(
            val_result["spatial_logits"],
            val_result["spectral_logits"],
            val_result["targets"],
            seed=seed,
        )
        started = time.perf_counter()
        variants = apply_ssrc_policy(
            policy, test_result["spatial_logits"], test_result["spectral_logits"]
        )
        selector_policy_seconds = time.perf_counter() - started
        spatial_calibrated = test_result["spatial_logits"] / policy.spatial_temperature
        spectral_calibrated = test_result["spectral_logits"] / policy.spectral_temperature
        global_predictions = variants["replay_global_logit_v5"][0].argmax(axis=1)
        for variant in pending:
            logits, weights, selector_scores = variants[variant]
            metrics, confusion, predictions = _variant_metrics(
                variant=variant,
                seed=seed,
                dataset=args.dataset,
                protocol=args.protocol,
                logits=logits,
                weights=weights,
                selector_scores=selector_scores,
                targets=test_result["targets"],
                spatial_logits=spatial_calibrated,
                spectral_logits=spectral_calibrated,
                global_predictions=global_predictions,
                policy=policy,
                source_inference_seconds=inference_seconds,
                selector_policy_seconds=selector_policy_seconds,
                source_parameter_count=parameter_count,
            )
            extra_arrays = (
                {"selector_scores.npy": np.asarray(selector_scores, dtype=np.float32)}
                if selector_scores is not None
                else None
            )
            _save_variant(
                run_dir=run_dirs[variant],
                metrics=metrics,
                confusion=confusion,
                predictions=predictions,
                logits=logits,
                weights=weights,
                targets=test_result["targets"],
                policy=policy,
                source_run=source_run,
                source_config=source_config,
                extra_arrays=extra_arrays,
            )
            print(
                f"SAVED {variant} seed={seed} oa={metrics['oa']:.6f} "
                f"switches={metrics['test_switch_count']}"
            )

    if args.no_aggregate:
        return
    report_dir = project_root / "reports" / args.output_experiment
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "aggregate.py"),
            "--results-root",
            str(output_root),
            "--output-dir",
            str(report_dir),
            "--reference-model",
            "replay_spatial_logit_v5,replay_global_logit_v5,replay_ssrc_score_v5",
        ],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "evaluate_ssrc_replay.py"),
            "--per-run",
            str(report_dir / "per_run.csv"),
            "--output-dir",
            str(report_dir),
        ],
        cwd=project_root,
        check=True,
    )
    print(f"REPORT {report_dir / 'ssrc_development_decision.md'}")


if __name__ == "__main__":
    main()
