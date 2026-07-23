# Research Proposal: Spatial-Anchored Selective Spectral Correction

## Problem Anchor

- Bottom-line problem: 在 spatial-block 小样本高光谱分类中，将空间与光谱分支的互补信息转化为稳定 OA 收益。
- Must-solve bottleneck: Pavia 的 oracle 相对 spatial 有 `9.816 pp` 空间，但 global fusion 损失 `3.842 pp` OA，ADLF 无收益；需要识别 spectral-only 正确样本，同时保护 spatial 已正确样本。
- Non-goals: 不增加新 backbone、attention、Mamba 或端到端大 gate；不使用测试标签训练或选阈值；不手写 Pavia 类别 2/5/6 规则。
- Constraints: NVIDIA L4；每类 30 train、10 validation；5 seeds；Pavia 已是开发数据；目标为可复现的 SCI 四区方案。
- Success condition: selector 以 spatial 为默认，仅在训练信息证明 spectral correction 有正净收益时切换，并在未查看数据集上稳定超过 spatial。

## Technical Gap

现有 soft fusion 会修改所有样本，而审计显示 spectral 独占正确集中在少数条件，spatial 独占正确总量约为 spectral 的 2.7 倍。单一 margin AUC 仅 `0.567`，无法表达类别条件可靠性。

## Method Thesis

学习每次 spectral hard correction 相对 spatial 的条件效用 `+1/0/-1`，并用风险下界控制 correction coverage；证据不足时严格 abstain。

## Proposed Method

1. 对训练区域执行内部 OOF backbone 训练，产生未见标签样本的空间/光谱 logits。
2. 特征只含置信度差、margin 差、entropy 差、JS divergence 及两个预测类别 one-hot。
3. 以 `spectral_correct - spatial_correct` 为 utility，拟合 L2 ridge selector。
4. 在 OOF 分数上选择 correction coverage，要求 beneficial rate 的单侧 Wilson 下界大于 0.5。
5. 推理时仅在分支不一致且 score 超阈值时使用 spectral logits，否则使用 spatial logits。

## Complexity Budget

- Dominant contribution: risk-controlled spatial-anchored correction。
- New trainable component: 一个低容量线性 utility selector。
- Excluded: class-pair interaction table、MLP router、test-time fitting、软 logit 混合。

## Validation

- Pavia 用于开发和删除检查。
- score-only 是类别特征消融，spatial/global 是主要参考。
- 通过后在 Houston 做首次确认，再扩展 KSC/Botswana。

## Initial Risk

直接进行训练区域 OOF 需要每 seed 多次重训 backbone；在 selector 本身尚未证明有效前，计算成本和工程复杂度偏高。
