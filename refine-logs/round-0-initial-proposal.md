# Research Proposal: Anchored Disagreement Logit Fusion for Leakage-Aware HSI Classification

## Problem Anchor

- Bottom-line problem: 在高光谱小样本 spatial-block 协议下，让空间与光谱分支的互补信息稳定转化为最终分类收益，而不是仅产生看似合理但与最终 OA 脱节的动态权重。
- Must-solve bottleneck: 当前 entropy gate 由辅助分支 logits 决定，却在特征层融合并交给独立分类器，导致路由质量与最终预测不一致；同时每类仅 10 个验证样本使温度和动态权重不稳定。
- Non-goals: 不继续包装 entropy-softmax；不堆叠更多注意力或 Mamba 模块；不使用测试标签选择融合参数；Pavia pilot 未通过前不扩展四数据集正式矩阵。
- Constraints: 复用现有 calibrated-v3 checkpoint 和不可变 split；Colab NVIDIA L4 22.5 GB；5 个固定 seeds；训练集每类 30、验证集每类 10；目标为严谨且可复现的 SCI 四区论文方案。
- Success condition: 仅使用训练/验证信息选择融合参数；候选相对 global-logit 与 spatial-only 平均 OA 均至少提高 1.0 pp、至少 4/5 seeds 为正、最坏退化不超过 1.5 pp，且 ECE/Brier 平均恶化不超过 0.01。

## Technical Gap

第三轮用辅助分支熵给特征加权，但最终类别由另一个融合分类器决定，因此路由诊断与最终 OA 没有同一决策语义。最小修复不是增加更大的 gate，而是让权重直接作用于两个分支的类别 logits，并用全局最优混合作为稳定锚点。每样本路由只能在分支不一致时做有界修正。

## Method Thesis

- One-sentence thesis: 用验证集确定的全局 logit 混合权重作为锚点，仅在分支预测不一致时依据置信间隔差做有界残差调整，使路由分数与最终类别决策处于同一 logit 空间。
- Smallest adequate intervention: 不重训编码器、不增加神经 gate，只增加一个全局标量和一个从四个固定候选中选择的残差半径。
- Frontier fit: 本问题没有自然的基础模型接口；强行加入 VLM/扩散模型会扩大贡献并破坏小样本可解释性，因此刻意采用透明的后验决策策略。

## Contribution Focus

- Dominant contribution: 风险受控、决策对齐的 disagreement-only logit routing。
- Supporting contribution: spatial-block 不可变 split、验证选参和可复算 artifact。
- Explicit non-contributions: 不声称新 backbone、KAN 或官方 Mamba；replay 未通过前不声称新融合方法有效。

## Proposed Method

### Complexity Budget

- Frozen/reused: `lassf_mlp_concat_norm_v3_h64` 的现有 checkpoint、split、预处理和两个辅助分类头。
- New parameters: 全局权重 `alpha0`、残差半径 `rho`、输出温度 `T`。
- Excluded: class-wise gate、MLP router、边界真值输入、测试集扫描、多源 checkpoint 选择。

### System Overview

```text
checkpoint + immutable split
  -> validation/test branch logits
  -> validation-only branch temperature scaling
  -> alpha0 selected by validation NLL
  -> disagreement confidence residual bounded by rho <= 0.15
  -> direct logit mixture
  -> validation-only final temperature
  -> one-shot test metrics and immutable artifacts
```

### Core Mechanism

设校准后的空间和光谱 logits 为 `zs` 与 `zp`。验证集在固定网格上选择 `alpha0`，最小化 `NLL(alpha*zs + (1-alpha)*zp)`。分支一致时 `alpha(x)=alpha0`；分支不一致时，以两分支 top-1/top-2 概率间隔差 `d(x)` 构造：

```text
alpha(x) = clip(alpha0 + rho * tanh(d(x) / sigma_val), 0, 1)
z(x) = alpha(x) * zs(x) + (1-alpha(x)) * zp(x)
```

`rho` 只从 `{0, 0.05, 0.10, 0.15}` 中按验证 NLL 选择；并列时选择更小值。`rho=0` 精确退化为 global-logit，限制动态路由的最坏风险。最终温度只在验证集拟合。

### Training and Inference

本轮不训练 backbone。每个 seed 独立恢复固定来源 checkpoint，重新生成 validation/test logits。所有温度、`alpha0`、`rho` 和尺度统计只由 validation 计算；test 只调用一次确定后的规则。保存参数、验证目标、测试预测、混合 logits、权重、指标和源 checkpoint 哈希。

### Failure Modes and Diagnostics

- `rho=0` 占多数 seeds: 动态路由没有额外价值，停止该主线。
- routing AUC 高但 OA 低: 检查 margin 方向、最终温度及分歧样本净收益。
- 单 seed 大幅退化: 由 `rho<=0.15` 限制，并执行最坏退化阈值。
- 验证过拟合: 固定极小网格、记录自由度；若 pilot 通过，正式阶段改用训练集交叉拟合。

## Novelty and Elegance Argument

核心不是普通置信度加权，而是把第三轮暴露的“辅助路由与最终分类头错位”修正为同一 logit 决策，并以 global fusion 作为显式收缩先验。方法只有一个动态自由度，能够被删除检查直接证伪。正式写作前仍需对 confidence-aware/logit routing HSI 文献查新。

## Claim-Driven Validation Sketch

### Claim 1: 决策对齐优于特征级 entropy routing

- Baselines: spatial-logit、spectral-logit、mean-logit、global-logit、第三轮 entropy-softmax。
- Candidate: anchored disagreement logit fusion (ADLF)。
- Evidence: 相对 global-logit 和 spatial-logit 的配对 OA、最坏 seed、ECE/Brier。

### Claim 2: 有界残差提供稳定性

- Ablation: `rho=0` global-logit 与验证选择的 `rho<=0.15`。
- Evidence: 4/5 seeds 为正且最坏退化不超过 1.5 pp；若失败立即停止。

## Experiment Handoff Inputs

- Source model fixed before replay: `lassf_mlp_concat_norm_v3_h64`。
- Dataset/protocol: Pavia University / spatial-block / seeds 0-4。
- Output variants: spatial, spectral, mean, global, ADLF。
- Decision: ADLF 相对 spatial/global 均满足 Anchor 中全部阈值才扩展。

## Compute & Timeline Estimate

- GPU: 只做 5 个 checkpoint 的 validation/test 推理，预计 L4 小于 0.5 GPU-hour。
- Data cost: 无新增标注。
- Timeline: 实现与测试 1 个工作日；Colab replay 约数十分钟。
