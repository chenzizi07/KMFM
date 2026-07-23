import pandas as pd

from scripts.evaluate_oasd import BASELINE, CANDIDATE, UNIFORM, evaluate_group


def test_oasd_decision_applies_all_fixed_checks_without_torch():
    rows = []
    for seed in range(5):
        rows.extend(
            [
                {
                    "model": CANDIDATE,
                    "seed": seed,
                    "oa": 0.72,
                    "aa": 0.71,
                    "ece": 0.04,
                    "brier": 0.10,
                    "distillation_active_class_count": 2,
                },
                {
                    "model": BASELINE,
                    "seed": seed,
                    "oa": 0.71,
                    "aa": 0.70,
                    "ece": 0.05,
                    "brier": 0.11,
                    "distillation_active_class_count": 0,
                },
                {
                    "model": UNIFORM,
                    "seed": seed,
                    "oa": 0.715,
                    "aa": 0.705,
                    "ece": 0.045,
                    "brier": 0.105,
                    "distillation_active_class_count": 2,
                },
            ]
        )
    decision = evaluate_group(pd.DataFrame(rows))
    assert decision["decision"] == "DEVELOPMENT_GO"
    assert decision["checks_passed"] == decision["checks_total"]
