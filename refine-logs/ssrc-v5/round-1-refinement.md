# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 在 spatial-block 小样本高光谱分类中，将空间与光谱分支的互补信息转化为稳定 OA 收益。
- Must-solve bottleneck: Pavia 的 oracle 相对 spatial 有 `9.816 pp` 空间，但 global fusion 损失 `3.842 pp` OA，ADLF 无收益；需要识别 spectral-only 正确样本，同时保护 spatial 已正确样本。
- Non-goals: 不增加新 backbone、attention、Mamba 或端到端大 gate；不使用测试标签训练或选阈值；不手写 Pavia 类别 2/5/6 规则。
- Constraints: NVIDIA L4；每类 30 train、10 validation；5 seeds；Pavia 已是开发数据；目标为可复现的 SCI 四区方案。
- Success condition: selector 以 spatial 为默认，仅在训练信息证明 spectral correction 有正净收益时切换，并在未查看数据集上稳定超过 spatial。

## Anchor Check

- Original bottleneck: soft fusion and scalar confidence cannot protect spatial-correct samples while exploiting conditional spectral wins.
- Preservation: the revised policy predicts correction utility directly and abstains by default.
- Rejected drift: no new backbone, no manually selected class list, and no test-derived threshold.

## Simplicity Check

- Dominant contribution: risk-controlled conditional hard correction.
- Removed from the immediate stage: internal backbone retraining and class-pair interactions.
- New component count: one ridge selector.
- Development variants: score-only deletion check and class-aware candidate.

## Changes Made

1. Added a checkpoint-only development gate before expensive train-region OOF.
2. Made temperature fitting fold-specific when producing validation OOF selector scores.
3. Fixed ridge alpha, features, coverage grid, confidence level, and minimum exclusive count before test replay.
4. Reserved formal train-region OOF and Houston confirmation for a successful development decision.

## Revised Proposal

### Technical Gap

The measured oracle gap is large in every seed, but spectral wins are conditional and asymmetric. Global/soft mixing improves macro AA while lowering OA, and the support-gain Spearman is negative. A useful selector must estimate sample-level correction utility while respecting the spatial branch as the safer default.

### Method Thesis

SSRC estimates the expected correctness change caused by replacing the spatial decision with the spectral decision, then permits only corrections whose cross-fitted validation evidence has a positive risk lower bound.

### Development Mechanism

For calibrated branch logits, construct four continuous features:

```text
spectral_confidence - spatial_confidence
spectral_margin - spatial_margin
spatial_entropy - spectral_entropy
Jensen-Shannon(spatial_probability, spectral_probability)
```

The class-aware candidate adds separate one-hot vectors for the spatial and spectral predicted classes. It never receives the true class. The utility target is:

```text
u(x) = I[spectral correct] - I[spatial correct] in {-1, 0, +1}
```

A ridge regressor with fixed `alpha=10` predicts utility only for branch-disagreement samples. Five stratified validation folds produce OOF scores; each fold fits its own two branch temperatures and selector on the other folds.

### Risk Control

Candidate correction coverages are fixed at `{5%, 10%, 20%, 30%}` of disagreement samples. A candidate is admissible only when:

- at least five OOF samples have non-zero utility;
- net corrected errors are positive;
- the one-sided 80% Wilson lower bound for beneficial corrections exceeds 0.5.

Among admissible candidates, select maximum net corrections, then higher lower bound, then fewer switches. If no candidate is admissible, use no corrections.

### Inference

Fit final temperatures and ridge selector on all validation samples. At test time:

```text
if branch predictions disagree and utility_score >= threshold
   and the frozen disagreement-coverage budget is not exhausted:
    output spectral logits
else:
    output spatial logits
```

No test labels, test-time optimization, or soft mixture are used. The validation-fitted
temperatures, selector, threshold, and coverage are frozen before test inference.

### Fixed Variants

1. spatial-logit;
2. spectral-logit;
3. global-logit;
4. SSRC score-only;
5. SSRC class-aware candidate.

### Development Decision

The class-aware candidate must beat spatial and global by at least `1.0 pp` mean OA with `4/5` positive seeds and no seed below `-1.0 pp`. It must beat score-only by `0.5 pp` with `3/5` positives, activate in `3/5` seeds, recover at least 15% of the oracle gap, and keep ECE/Brier degradation versus spatial below `0.01`.

### Formal Transition

A Pavia `DEVELOPMENT_GO` authorizes train-region OOF backbone supervision. Houston remains the first unseen confirmation dataset. Failure terminates learned selector development rather than triggering another gate search.

### Compute

The development replay restores five existing checkpoints and is expected below 0.5 L4 GPU-hour. Formal OOF training is deferred until this gate passes.
