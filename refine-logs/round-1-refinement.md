# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 在高光谱小样本 spatial-block 协议下，让空间与光谱分支的互补信息稳定转化为最终分类收益，而不是仅产生看似合理但与最终 OA 脱节的动态权重。
- Must-solve bottleneck: 当前 entropy gate 由辅助分支 logits 决定，却在特征层融合并交给独立分类器，导致路由质量与最终预测不一致；同时每类仅 10 个验证样本使温度和动态权重不稳定。
- Non-goals: 不继续包装 entropy-softmax；不堆叠更多注意力或 Mamba 模块；不使用测试标签选择融合参数；Pavia pilot 未通过前不扩展四数据集正式矩阵。
- Constraints: 复用现有 calibrated-v3 checkpoint 和不可变 split；Colab NVIDIA L4 22.5 GB；5 个固定 seeds；训练集每类 30、验证集每类 10；目标为严谨且可复现的 SCI 四区论文方案。
- Success condition: 仅使用训练/验证信息选择融合参数；候选相对 global-logit 与 spatial-only 平均 OA 均至少提高 1.0 pp、至少 4/5 seeds 为正、最坏退化不超过 1.5 pp，且 ECE/Brier 平均恶化不超过 0.01。

## Anchor Check

- Original bottleneck: feature-level entropy routing does not control the final classifier decision.
- Preservation: every new weight now acts directly on branch class logits and is fitted without test labels.
- Rejected drift: foundation models, new backbones, class-wise gates, and boundary labels are excluded.

## Simplicity Check

- Dominant contribution: global-anchored, disagreement-only, bounded logit routing.
- Removed: final output temperature, class-wise shrinkage, trainable MLP router, multi-checkpoint selection.
- Remaining mechanism: two branch temperatures, one global alpha, one radius chosen from four fixed values.

## Changes Made

1. Removed final output temperature to reduce validation degrees of freedom.
2. Fixed source model to `lassf_mlp_concat_norm_v3_h64` before replay.
3. Added disagreement-set net gain, improved/harmed counts, and fallback-rate diagnostics.
4. Reframed replay as a feasibility gate; passing it does not by itself establish manuscript novelty.

## Revised Proposal

### Technical Gap

Third-round entropy estimates came from auxiliary heads, weighted features, and then passed through an unrelated fusion classifier. The route score and final prediction therefore had different semantics. The smallest correction is to route in the class-decision space and shrink every sample-specific adjustment toward a stable global mixture.

### Method Thesis

For calibrated branch logits `zs` and `zp`, select a global spatial weight `alpha0` on validation NLL. Only when branch top-1 predictions disagree, adjust that weight by a bounded residual derived from their confidence-margin difference:

```text
d(x) = margin_softmax(zs) - margin_softmax(zp)
r(x) = 0                                      if argmax(zs) == argmax(zp)
       tanh(d(x) / sigma_validation)          otherwise
alpha(x) = clip(alpha0 + rho * r(x), 0, 1)
z(x) = alpha(x) * zs(x) + (1-alpha(x)) * zp(x)
```

`alpha0` is selected from `{0.00, 0.05, ..., 1.00}` and `rho` from `{0, 0.05, 0.10, 0.15}` by validation NLL, with smaller values winning ties. Each branch temperature is fitted independently on validation once and reused by every variant. There is no final temperature.

### Complexity Budget

- Reused: existing concat-normalized checkpoint, immutable split, loaders, auxiliary heads.
- New trainable neural components: zero.
- Fitted scalar quantities: two temperatures, `alpha0`, `rho`, validation score scale.
- Explicitly excluded: test-time fitting, class-specific weights, learned router, ground-truth boundary features.

### Replay Protocol

1. Verify source `status.json`, resolved config, split, and checkpoint.
2. Recreate train-support normalization and validation/test loaders.
3. Restore checkpoint with branch temperatures reset to 1.
4. Run validation once, fit branch temperatures, global alpha, and bounded radius.
5. Freeze all choices, run test once, and evaluate five variants: spatial, spectral, mean, global, ADLF.
6. Save predictions, logits, weights, selected parameters, metrics, source hashes, and a run status.
7. Aggregate five seeds and apply the pre-registered decision.

### Claim-Driven Validation

- Main claim: decision-aligned ADLF converts branch complementarity more reliably than feature entropy routing.
- References: calibrated spatial logits and validation-fitted global logit fusion.
- Mandatory checks for each reference: mean OA gain >= 1.0 pp, >=4/5 positive pairs, worst loss >= -1.5 pp.
- Calibration checks vs global: mean ECE and Brier degradation <= 0.01.
- Mechanism diagnostics: disagreement routing AUC, disagreement-set net corrected errors, `rho=0` frequency.

### Failure Decision

- If fewer than 3/5 seeds choose `rho>0`, dynamic routing lacks repeatable value.
- If any mandatory performance check fails, stop ADLF and retain global fusion only as a baseline.
- If replay passes, train a neutral dual-branch model with cross-fitted router supervision before four-dataset experiments.

### Compute

Five checkpoint replays only; expected below 0.5 L4 GPU-hour and no new labels.
