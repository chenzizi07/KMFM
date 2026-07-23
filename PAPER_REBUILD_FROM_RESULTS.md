# 从真实实验重构 SCI 四区论文

## 建议题目

在真实结果支持前，只使用工作题目：

> Leakage-Aware Reliability Fusion for Hyperspectral Image Classification under Spatially Disjoint Evaluation

不要继续把 KAN 或 Mamba 放入标题。当前实现是 pure-PyTorch selective SSM；KAN 也不是默认模型。

## 唯一主贡献

> 在空间互斥 HSI 分类中，依据空间/光谱分支的预测熵与特征分歧进行可靠性路由，降低不可靠空间上下文的影响。

支撑贡献只保留一个：给出 random-pixel 与 spatial-block 的受控对比和可复算实验协议。

## 论文结构

### Introduction

1. 传统 random-pixel 评估容易利用空间自相关；
2. 空间上下文不是在所有位置都可靠，静态融合不能处理这种差异；
3. 提出 leakage-aware protocol 与 reliability fusion；
4. 强调结果和代码的可追溯性。

### Related Work

- HSI CNN/Transformer/SSM；
- spectral–spatial fusion；
- spatially disjoint HSI evaluation；
- uncertainty-aware routing。

不要把 Related Work 写成 KAN 与 Mamba 的模块清单。正式写稿前重新查新 2026 年相关文献。

### Method

只解释：

1. patch 与 context boundary；
2. spectral encoder；
3. spatial selective SSM；
4. auxiliary entropy；
5. reliability gate 和损失。

### Experiments

主表优先使用 SB；RP 放为兼容对照。至少包含：

- 四个数据集及文件哈希/版本；
- 固定 mask 和每类样本数；
- 四个核心模型；
- OA/AA/Kappa、95% CI、配对检验；
- 参数量、训练时间、测试时间；
- gate 分布、边界与内部区域分析。

### Discussion

- 解释 RP–SB 性能差距；
- 分析 gate 是否在空间分支高熵时降低空间权重；
- 说明 patch/block 设置的局限；
- 不把单场景结果扩大为跨传感器泛化。

## Go/No-Go 决策

完成两个数据集、两个 seeds 的 pilot 后：

1. 主模型若在 SB 上不优于 concat/plain gate，停止“可靠性融合”论文主线；
2. Conv1d 若不优于 MLP，保留 MLP 并删除谱算子贡献；
3. RP 很高但 SB 明显下降时，如实把论文改为协议诊断；
4. 只有所有数字能通过 `verify_run.py`，才进入论文表格。

## SCI 四区可行性

如果真实实验满足以下条件，四区应用型遥感/传感方向具有合理可行性：

- 四个数据集至少三个支持核心趋势；
- SB 主结果完整，且不是只报告 RP；
- 有参数匹配的 concat/plain gate 对照；
- 结果包含逐 seed 和统计检验；
- 代码与 masks 能复现至少一个完整表格。

这不是录用保证。期刊分区和收录状态会变化，投稿前必须按当年最新版核对。
