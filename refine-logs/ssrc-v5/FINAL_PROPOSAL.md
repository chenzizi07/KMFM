# Research Proposal: Risk-Controlled Spatial-Anchored Spectral Correction

## Problem Anchor

在 spatial-block 小样本高光谱分类中，将空间与光谱分支的互补信息转化为稳定 OA 收益。Pavia 的 oracle 相对 spatial 有 `9.816 pp` 空间，但 global fusion 损失 `3.842 pp` OA，ADLF 无收益。方法必须识别 spectral-only 正确样本，同时保护 spatial 已正确样本。

不增加新 backbone、attention、Mamba 或大规模神经 gate；不使用测试标签训练或选阈值；不把审计观察到的 Pavia 类别写成规则。

## Method Thesis

SSRC 直接预测“从 spatial 切换到 spectral”造成的正确性变化，并只允许 cross-fitted 风险下界为正的 hard correction；证据不足时保持 spatial。

## Development Method

### Inputs

复用 `lassf_mlp_concat_norm_v3_h64` checkpoint 的两个辅助分支 logits。Pavia 阶段不重新训练 backbone。

### Features

- spectral-spatial maximum-confidence difference;
- spectral-spatial top-two margin difference;
- spatial-spectral normalized-entropy difference;
- branch Jensen-Shannon divergence;
- candidate only: separate spatial/spectral predicted-class one-hot vectors。

真实类别只用于构造 validation utility，不作为输入。类别特征不包含 pair interaction table。

### Utility Model

```text
u(x) = I[spectral correct] - I[spatial correct]
```

`u=+1` 表示 spectral 可以纠正 spatial，`u=-1` 表示切换会伤害，`u=0` 表示切换不改变正确性。固定使用 `alpha=10` 的 ridge regression；score-only 版本是不使用类别 one-hot 的删除消融。

### Cross-Fitted Calibration

validation 按真实类别做 5 折。每一折只用其他四折拟合两个 branch temperature、特征标准化和 ridge selector，然后为留出折产生 OOF utility score。该步骤避免用同一样本同时拟合 selector 和评价其风险。

### Risk-Controlled Coverage

只评估 `{5%, 10%, 20%, 30%}` 四个固定 disagreement coverage。候选需要至少五个非零 utility 样本、正净纠错数，并且 beneficial correction rate 的单侧 80% Wilson 下界大于 0.5。满足条件的候选按净纠错数、下界、较少切换依次选择；没有安全候选时 threshold 为 `None`，精确退化为 spatial。

### Test Inference

在完整 validation 上拟合最终温度和 selector，并按已选 OOF coverage 将数值阈值对齐到
最终 selector 的分数尺度。测试时只执行：

```text
eligible = branch disagreement AND utility_score >= frozen threshold
spectral switch = top eligible scores, capped by frozen disagreement coverage
prediction logits = spectral logits if switch else spatial logits
```

coverage cap 只使用测试分数的排序和 disagreement 数量，不使用测试标签。

## Fixed Development Variants

1. `replay_spatial_logit_v5`
2. `replay_spectral_logit_v5`
3. `replay_global_logit_v5`
4. `replay_ssrc_score_v5`
5. `replay_ssrc_class_v5`

## Decision Boundary

- candidate 相对 spatial/global 平均 OA 均至少 `+1.0 pp`、至少 `4/5` seeds 为正；
- candidate 相对 score-only 平均 OA 至少 `+0.5 pp`、至少 `3/5` 为正；
- 相对三个 reference 的最坏退化不超过 `1.0 pp`；
- 至少 `3/5` seeds 启用 correction；
- 平均 oracle gap recovery 至少 `15%`；
- 相对 spatial 的 ECE/Brier 平均恶化不超过 `0.01`。

## Evidence Boundary

Pavia 已参与方法诊断，本轮结果只能决定是否继续开发。`DEVELOPMENT_GO` 后，正式方法必须使用训练区域 OOF backbone predictions 生成 selector supervision，并在未查看的 Houston 上首次确认；随后才扩展 KSC 和 Botswana。Pavia `DEVELOPMENT_NO_GO` 将终止 selector 路线。

## Contribution Boundary

唯一主贡献是风险控制的 spatial-anchored correction，不把 backbone、类别审计或实验协议包装成并列创新。论文创新性仍需在方法通过后针对 selective prediction、classifier routing 和 HSI decision fusion 正式查新。
