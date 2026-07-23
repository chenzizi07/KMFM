# 第六轮开发实验：OASD

OASD（Out-of-fold Advantage-weighted Spectral Distillation）终止测试时 selector，
仅在训练阶段使用光谱分支。每个 seed 的训练中心按类别分成固定 3 折，用未见过该中心
标签的折外预测估计类别级光谱优势；正优势经过固定收缩后成为蒸馏权重。最终推理只使用
空间分支，不产生测试时路由风险。

Pavia 已用于前五轮诊断，所以本轮仍是开发实验，不是论文结论。三个变体共享完全相同的
空间 MLP/SSM 架构、初始化规则和训练参数：

1. `lassf_mlp_spatial_only_v6_h64`：无蒸馏基线；
2. `lassf_mlp_uniform_distill_v6_h64`：所有类别均匀蒸馏对照；
3. `lassf_mlp_oof_adv_distill_v6_h64`：固定 OOF 类别优势蒸馏。

在已有 Colab notebook 的新代码单元运行：

```python
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh
!python scripts/run_colab_experiment.py \
  --dataset pavia_university \
  --protocols spatial_block \
  --suite oasd_v6 \
  --experiment pavia_oasd_dev_v6 \
  --seeds 0,1,2,3,4 \
  --epochs 120 \
  --patience 25 \
  --recover-incomplete
!cat reports/pavia_oasd_dev_v6/oasd_development_decision.md
```

固定判定只比较 5 个配对 seed：OASD 对空间基线平均 OA 至少 `+0.5 pp`、至少
`3/5` 为正、最差不低于 `-2 pp`；对均匀蒸馏平均不差且至少 `3/5` 为正；同时
AA 不下降、ECE/Brier 增量不超过 `0.015`，并且至少 3 个 seed 学到非零类别权重。

只有 `DEVELOPMENT_GO` 才冻结全部规则，在未查看的 Houston2013 上进行首次确认。
`DEVELOPMENT_NO_GO` 将终止基于双分支互补的主线，转向单分支空间 SSM 的稳健训练。
