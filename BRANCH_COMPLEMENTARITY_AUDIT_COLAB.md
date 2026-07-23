# 分支互补性审计

该步骤只分析第四轮已经保存的预测，不重新训练模型。测试标签仅用于测量
oracle 上限、类别级独占正确样本和现有融合的恢复率，不用于选择第五轮参数。

## Colab 命令

```bash
%cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
!bash scripts/colab_update.sh

!python scripts/audit_branch_complementarity.py \
  --experiment pavia_adlf_replay_v4 \
  --dataset pavia_university \
  --protocol spatial_block \
  --seeds 0,1,2,3,4
```

## 查看报告

```bash
!cat /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/pavia_adlf_replay_v4/complementarity_audit/complementarity_audit.md
```

主要输出：

- `complementarity_per_seed.csv`：每个 seed 的 spatial、spectral、global、ADLF、oracle 结果；
- `complementarity_per_class_seed.csv`：每个 seed、每个类别的独占正确和实际纠错；
- `complementarity_per_class_summary.csv`：五个 seed 的类别级汇总；
- `complementarity_audit.json`：决策、阈值、输入文件 SHA-256 和运行环境；
- `complementarity_audit.md`：可直接阅读的研究决策报告。

## 决策边界

- 平均 oracle OA 增益至少 `3 pp`，且至少 80% seeds 为正：只允许继续开发 selector；
- global OA 平均不优于 spatial：global fusion 只能保留为 baseline；
- ADLF 相对 global 平均不提升：当前 ADLF 路由正式终止；
- 至少 3 个 seeds 出现 global AA 提升但 OA 下降：进入类别风险错配分析。

上述判定只决定是否值得开发 selector，不构成论文有效性结论。Pavia 已是开发数据，
后续 selector 必须用训练区域 out-of-fold 监督，并在未查看过结果的数据集上首次确认。
