# 第四轮：ADLF Checkpoint Replay

本轮不重新训练编码器。脚本固定读取第三轮
`lassf_mlp_concat_norm_v3_h64` checkpoint，用 validation split 选择分支温度、
global logit 权重和最大 0.15 的 disagreement residual，然后一次性评估 test split。

## Colab 命令

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh

!python scripts/replay_logit_fusion.py \
  --source-experiment pavia_calibrated_v3 \
  --output-experiment pavia_adlf_replay_v4 \
  --dataset pavia_university \
  --protocol spatial_block \
  --source-model lassf_mlp_concat_norm_v3_h64 \
  --seeds 0,1,2,3,4
```

源实验的 5 个 concat-normalized runs 必须均为 `success`，且包含
`checkpoint_best.pt`、`resolved_config.json` 和对应 immutable split。

## 固定 replay 变体

1. `replay_spatial_logit_v4`
2. `replay_spectral_logit_v4`
3. `replay_mean_logit_v4`
4. `replay_global_logit_v4`
5. `replay_adlf_v4`

## 输出

```bash
!cat /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/pavia_adlf_replay_v4/adlf_replay_decision.md
```

主要文件：

- `summary.csv`: 五个 replay 变体的 5-seed 汇总；
- `per_run.csv`: 每个 seed 的选定 alpha、radius、验证 NLL 和分歧诊断；
- `paired_tests.csv`: ADLF/global/spatial 的配对比较；
- `adlf_replay_decision.md` / `.json`: 预注册 Go/No-Go 判定。

## 中断恢复

确认旧单元已停止后，在相同命令末尾添加：

```bash
  --recover-incomplete
```

成功的 replay run 会跳过；未完成目录先归档到 `results/_incomplete` 后重算。

## Go/No-Go

ADLF 必须同时满足：

- 相对 calibrated spatial-logit 和 global-logit 平均 OA 均至少提高 1.0 pp；
- 相对每个参考至少 4/5 seeds 为正；
- 任一 seed 相对每个参考退化不超过 1.5 pp；
- 相对 global-logit 的平均 ECE/Brier 恶化均不超过 0.01；
- 至少 3/5 seeds 在 validation 上选择非零 residual radius。

Routing AUC 只作为诊断，不再单独作为通过条件。
