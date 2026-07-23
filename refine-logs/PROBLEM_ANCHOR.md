# Problem Anchor

- Bottom-line problem: 在高光谱小样本 spatial-block 协议下，让空间与光谱分支的互补信息稳定转化为最终分类收益，而不是仅产生看似合理但与最终 OA 脱节的动态权重。
- Must-solve bottleneck: 当前 entropy gate 由辅助分支 logits 决定，却在特征层融合并交给独立分类器，导致路由质量与最终预测不一致；同时每类仅 10 个验证样本使温度和动态权重不稳定。
- Non-goals: 不继续包装 entropy-softmax；不堆叠更多注意力或 Mamba 模块；不使用测试标签选择融合参数；Pavia pilot 未通过前不扩展四数据集正式矩阵。
- Constraints: 复用现有 calibrated-v3 checkpoint 和不可变 split；Colab NVIDIA L4 22.5 GB；5 个固定 seeds；训练集每类 30、验证集每类 10；目标为严谨且可复现的 SCI 四区论文方案。
- Success condition: 仅使用训练/验证信息选择融合参数；候选相对 global-logit 与 spatial-only 平均 OA 均至少提高 1.0 pp、至少 4/5 seeds 为正、最坏退化不超过 1.5 pp，且 ECE/Brier 平均恶化不超过 0.01。
