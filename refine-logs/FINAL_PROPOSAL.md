# Research Proposal: Anchored Disagreement Logit Fusion for Leakage-Aware HSI Classification

## Problem Anchor

- Bottom-line problem: 在高光谱小样本 spatial-block 协议下，让空间与光谱分支的互补信息稳定转化为最终分类收益，而不是仅产生看似合理但与最终 OA 脱节的动态权重。
- Must-solve bottleneck: 当前 entropy gate 由辅助分支 logits 决定，却在特征层融合并交给独立分类器，导致路由质量与最终预测不一致；同时每类仅 10 个验证样本使温度和动态权重不稳定。
- Non-goals: 不继续包装 entropy-softmax；不堆叠更多注意力或 Mamba 模块；不使用测试标签选择融合参数；Pavia pilot 未通过前不扩展四数据集正式矩阵。
- Constraints: 复用现有 calibrated-v3 checkpoint 和不可变 split；Colab NVIDIA L4 22.5 GB；5 个固定 seeds；训练集每类 30、验证集每类 10；目标为严谨且可复现的 SCI 四区论文方案。
- Success condition: 仅使用训练/验证信息选择融合参数；候选相对 global-logit 与 spatial-only 平均 OA 均至少提高 1.0 pp、至少 4/5 seeds 为正、最坏退化不超过 1.5 pp，且 ECE/Brier 平均恶化不超过 0.01。

## Technical Gap

第三轮方法用辅助分支 logits 的熵给空间与光谱特征加权，最终类别却由独立融合分类器决定。路由信号和分类结果因此不处于同一决策空间，较好的 routing diagnostic 不能保证较高 OA。最小修复是直接融合两个分支的类别 logits，并将样本级调整收缩到稳定的全局混合权重附近。

## Method Thesis

以验证集最优的全局 logit 混合权重为锚点，仅在两分支 top-1 预测不一致时，按校准置信间隔差做幅度不超过 0.15 的残差调整。权重直接决定最终类别 logits，不再经过无关的融合分类头。

这是针对现有失败点的最小充分机制：不重训编码器、不增加神经 gate、没有 class-wise 参数，也不加入与问题无直接关系的基础模型组件。

## Contribution Focus

- Dominant contribution: 风险受控、决策对齐的 disagreement-only logit routing。
- Supporting contribution: 使用不可变 spatial-block split、validation-only 选参、配对五种子判定和可追溯 artifact 的泄漏防护评估流程。
- Explicit non-contributions: 不声称新 backbone、KAN 或官方 Mamba；checkpoint replay 通过只代表可行性，不足以单独证明论文创新性。

## Proposed Method

### Complexity Budget

- Frozen/reused: `lassf_mlp_concat_norm_v3_h64` checkpoint、数据 split、标准化流程、空间和光谱辅助分类头。
- New trainable neural components: zero。
- Validation quantities: 两个分支温度、全局空间权重 `alpha0`、残差半径 `rho` 和 margin scale。
- Excluded: 最终输出温度、class-wise gate、MLP router、测试集扫描、边界真值输入和多 checkpoint 选择。

### System Overview

```text
fixed checkpoint + immutable split
  -> validation branch logits
  -> fit one temperature per branch on validation NLL
  -> select global alpha0 on a fixed validation grid
  -> select bounded disagreement radius rho on validation NLL
  -> freeze policy
  -> one test pass
  -> direct branch-logit fusion and immutable artifacts
```

### Core Mechanism

设温度校准后的空间和光谱 logits 为 `zs(x)` 与 `zp(x)`。先在固定网格 `{0.00, 0.05, ..., 1.00}` 上选择 `alpha0`：

```text
alpha0 = argmin_alpha NLL(alpha * zs + (1 - alpha) * zp)
```

以 softmax top-1 与 top-2 概率差定义每个分支的 margin，并计算：

```text
d(x) = margin(zs(x)) - margin(zp(x))
r(x) = 0                              if argmax(zs) == argmax(zp)
       tanh(d(x) / sigma_validation)  otherwise
alpha(x) = clip(alpha0 + rho * r(x), 0, 1)
z(x) = alpha(x) * zs(x) + (1 - alpha(x)) * zp(x)
```

`sigma_validation` 是验证集分歧样本 `d(x)` 的标准差，样本不足时回退到全部验证样本，并设下限 `1e-3`。`rho` 只从 `{0, 0.05, 0.10, 0.15}` 中按验证 NLL 选择；并列时选择较小值。`rho=0` 精确退化为 global-logit baseline。

### Replay Protocol

1. 检查来源 run 的 `status.json`、`resolved_config.json`、`checkpoint_best.pt` 和 split。
2. 按来源配置重建只由训练区域拟合的标准化与 validation/test loaders。
3. 恢复 checkpoint，并将模型内分支温度重置为 1，提取原始分支 logits。
4. 只用 validation 拟合两个温度、`alpha0`、`rho` 和尺度统计。
5. 冻结策略，对 test 进行一次推理并同时生成 spatial、spectral、mean、global 和 ADLF 五个变体。
6. 保存 logits、预测、权重、混淆矩阵、选定参数、环境、输入哈希、运行状态和聚合报告。

## Claim-Driven Validation

### Claim 1: 决策对齐能稳定利用分支互补性

- Candidate: `replay_adlf_v4`。
- References: calibrated spatial-logit 与 validation-fitted global-logit。
- Decisive evidence: 相对每个 reference 的平均 OA 增益至少 1.0 pp、至少 4/5 seeds 为正，任一 seed 最坏退化不超过 1.5 pp。

### Claim 2: 动态残差在校准和稳定性上有实际价值

- Deletion check: `rho=0` global-logit。
- Decisive evidence: 至少 3/5 seeds 在 validation 选择 `rho>0`；相对 global 的平均 ECE 和 Brier 恶化均不超过 0.01。
- Diagnostic only: routing AUC、分歧样本 corrected/harmed counts 和净纠错数不作为单独通过条件。

## Failure Decision

- 任一预注册条件失败：本轮判定 `NO_GO`，停止包装 ADLF，不扩展四数据集。
- 多数 seeds 选择 `rho=0`：保留 global logit fusion 作为基线，结论是样本级路由没有稳定价值。
- Pilot 全部通过：再训练不偏向 concat classifier 的中性双分支模型，使用训练集交叉拟合产生 router supervision，并在四个数据集上完成正式基线、消融、校准和复杂度矩阵。

## Novelty Boundary

ADLF 的潜在新意在于把辅助路由与最终分类器错位的问题改写为同一 logit 决策空间中的全局锚定、有界分歧路由。该叙事需要在 pilot 通过后再针对 HSI confidence-aware fusion、mixture-of-experts routing 和 logit ensembling 做正式查新。当前 replay 只回答机制是否值得继续，不预设创新性成立。

## Compute and Timeline

- Pilot compute: 五个现有 checkpoint 的 validation/test replay，预计低于 0.5 L4 GPU-hour。
- New annotation: none。
- Immediate milestone: 先输出 Pavia 五种子 `GO/NO_GO`；只有 `GO` 才进入四数据集正式研究阶段。
