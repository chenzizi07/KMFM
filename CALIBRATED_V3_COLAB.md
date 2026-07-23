# 第三轮校准融合机制实验

本轮只回答一个问题：在 Pavia University 的 spatial-block 小样本协议下，校准后的
entropy-softmax 是否稳定优于 spatial-only 和经过归一化的普通门控。

## Colab 运行

继续使用原来的 notebook。在完成 Drive 挂载后，新建一个代码单元运行：

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh

!python scripts/run_colab_experiment.py \
  --dataset pavia_university \
  --experiment pavia_calibrated_v3 \
  --suite calibrated_v3 \
  --seeds 0,1,2,3,4 \
  --protocols spatial_block \
  --epochs 100 --patience 20 \
  --batch-size 128 --num-workers 2
```

脚本会复用已有的不可变 split，不会覆盖前两轮结果。中断后再次运行相同命令时，已经
成功的 run 会被跳过；若某个 run 目录不完整，脚本会停止并要求检查，而不会静默覆盖。

## 固定模型矩阵

1. `lassf_mlp_spatial_only_v3_h64`
2. `lassf_mlp_spectral_only_v3_h64`
3. `lassf_mlp_concat_norm_v3_h64`
4. `lassf_mlp_global_norm_v3_h64`
5. `lassf_mlp_gate_norm_v3_h64`
6. `lassf_mlp_entropy_softmax_v3_h64`

所有模型使用相同 MLP 光谱编码器、空间编码器、分支适配器和 split。候选方法额外使用
验证集温度缩放和固定温度的 entropy-softmax，不使用高维 reliability gate。

## 自动输出

实验结束后查看：

```bash
!cat /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/pavia_calibrated_v3/mechanism_decision.md
```

报告目录包含：

- `per_run.csv`：逐 split 指标和机制诊断；
- `summary.csv`：均值、样本标准差、95% t 区间和最差 split；
- `paired_tests.csv`：相对 spatial-only 和普通门控的配对比较；
- `mechanism_decision.json` / `.md`：预先固定阈值的 Go/No-Go 判定。

每个 run 还会保存原始及校准 logits、两个分支预测、gate、entropy、特征范数、加权贡献
范数、contribution ratio、测试坐标和边界距离。

## Go/No-Go 规则

候选方法必须同时满足：

- 相对 spatial-only 和 normalized plain gate 的平均 OA 均至少提高 1.0 个百分点；
- 相对每个参考方法至少 4/5 splits 为正；
- 任一配对 split 的 OA 退化不超过 2.0 个百分点；
- 分支预测不一致样本上的平均 routing AUC 不低于 0.60；
- 相对普通门控的平均 ECE 和 Brier 恶化均不超过 0.01。

这是 pilot 决策条件，不是统计显著性声明。只有本轮为 `GO` 后，才在 KSC 复现实验；
否则停止 reliability-fusion 主线，不扩展四数据集正式矩阵。
