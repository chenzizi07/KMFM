import numpy as np

from kmfm.spectral_audit import (
    StabilityThresholds,
    evaluate_encoder_profiles,
    repeated_oof_stability_profile,
    select_encoder,
)


def _predictions_with_stable_advantage():
    targets = np.repeat(np.arange(3), 10)
    spatial = np.tile(targets, (3, 1))
    spectral = spatial.copy()
    for repeat in range(3):
        for class_id in range(3):
            selected = np.flatnonzero(targets == class_id)
            spatial[repeat, selected[:3]] = (class_id + 1) % 3
            spectral[repeat, selected[:3]] = class_id
    return targets, spatial, spectral


def test_repeated_profile_detects_repeat_stable_classes():
    targets, spatial, spectral = _predictions_with_stable_advantage()
    profile = repeated_oof_stability_profile(
        targets,
        spatial,
        spectral,
        num_classes=3,
        thresholds=StabilityThresholds(min_mean_advantage=0.2),
    )
    assert profile["stable_class_ids"] == [0, 1, 2]
    assert profile["stable_net_correct"] == 9
    assert profile["seed_qualified"] is True


def test_repeated_profile_rejects_unstable_single_repeat_gain():
    targets, spatial, spectral = _predictions_with_stable_advantage()
    spatial[1:] = targets
    spectral[1:] = targets
    profile = repeated_oof_stability_profile(targets, spatial, spectral, num_classes=3)
    assert profile["stable_class_count"] == 0
    assert profile["seed_qualified"] is False


def test_encoder_decision_requires_repeatable_seed_level_evidence():
    passing_profiles = [
        {
            "seed_qualified": seed < 4,
            "stable_class_count": 3 if seed < 4 else 1,
            "global_oa_gap": -0.02,
            "stable_net_correct": 4 if seed < 4 else -1,
        }
        for seed in range(5)
    ]
    passing = evaluate_encoder_profiles("conv1d", passing_profiles)
    failing = evaluate_encoder_profiles(
        "mlp",
        [
            {
                "seed_qualified": False,
                "stable_class_count": 1,
                "global_oa_gap": -0.08,
                "stable_net_correct": -1,
            }
            for _ in range(5)
        ],
    )
    decision = select_encoder([failing, passing])
    assert passing["passed"] is True
    assert decision["decision"] == "DEVELOPMENT_GO"
    assert decision["selected_encoder"] == "conv1d"
