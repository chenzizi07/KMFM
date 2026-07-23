# 第五轮开发实验：SSRC

SSRC（Spatial-Anchored Selective Spectral Correction）固定复用第三轮
`lassf_mlp_concat_norm_v3_h64` checkpoint，不重新训练 backbone。它用 validation
的 5 折 OOF utility score 选择风险阈值，测试时仅在证据充分且两分支预测不一致时
切换到 spectral，其余样本保持 spatial。测试切换数还受验证阶段冻结的 disagreement
coverage 上限约束，避免测试分数分布漂移造成过度切换。

Pavia 已被用于方向诊断，因此本轮只能作为 development gate，不能作为论文确认结果。

## Colab 命令

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh

!python scripts/replay_selective_correction.py \
  --source-experiment pavia_calibrated_v3 \
  --output-experiment pavia_ssrc_dev_v5 \
  --dataset pavia_university \
  --protocol spatial_block \
  --source-model lassf_mlp_concat_norm_v3_h64 \
  --seeds 0,1,2,3,4
```

## 固定变体

1. `replay_spatial_logit_v5`
2. `replay_spectral_logit_v5`
3. `replay_global_logit_v5`
4. `replay_ssrc_score_v5`：仅连续置信特征；
5. `replay_ssrc_class_v5`：连续特征加两个分支的预测类别，是固定候选。

连续特征只有四个：置信度差、margin 差、entropy 差和 Jensen-Shannon divergence。
类别特征只使用两个分支的预测类别 one-hot，不使用真实类别或手写类别规则。

## 查看结果

```bash
!cat /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/pavia_ssrc_dev_v5/ssrc_development_decision.md
```

完整报告位于：

```text
/content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/pavia_ssrc_dev_v5/
```

## 中断恢复

确认旧进程已停止后，在命令末尾添加：

```bash
  --recover-incomplete
```

成功 run 会跳过；未完成目录先归档到 `results/_incomplete`。

## Development Go/No-Go

候选必须同时满足：

- 相对 spatial 和 global 平均 OA 均至少 `+1.0 pp`，至少 `4/5` seeds 为正；
- 相对 score-only 平均 OA 至少 `+0.5 pp`，至少 `3/5` seeds 为正；
- 相对三个参考的最坏 seed 退化均不超过 `1.0 pp`；
- 至少 `3/5` seeds 实际启用 correction；
- 平均 oracle gap recovery 至少 `15%`；
- 相对 spatial 的平均 ECE/Brier 恶化均不超过 `0.01`。

只有 `DEVELOPMENT_GO` 才进入训练区域 OOF 和 Houston 首次确认；本轮即使通过也不是论文结论。
