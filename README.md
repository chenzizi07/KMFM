# KMFM Rebuild / LASSF

这是根据旧 KMFM 论文和代码审计结果重新实现的干净工程。主目标不是复刻旧论文中的不可追溯数字，而是用真实数据建立一条可在 Google Colab 运行、能够自动复算结果的实验链。

主方法暂称 **LASSF（Leakage-Aware Selective Spectral–Spatial Fusion）**：

- 正确沿波段轴运行的 spectral Conv1d；
- 纯 PyTorch 多方向 selective state-space scan；
- 使用空间/光谱分支熵与特征分歧的可靠性门控；
- random-pixel 和 spatial-block 两种明确分开的协议；
- 每个 run 保存 mask 哈希、配置、checkpoint、prediction、confusion matrix 和指标。

本工程不会生成或填充论文结果。能否发表 SCI 四区取决于真实实验是否支持主张。

## 1. 目录

```text
KMFM/
  configs/                  示例配置
  notebooks/                Colab notebook
  scripts/
    make_split.py           生成固定划分
    train.py                一个配置/数据集/协议/模型/seed 的训练
    verify_run.py           从预测重算混淆矩阵和指标
    aggregate.py            多 seed 汇总、置信区间和配对检验
    export_table.py         从汇总 CSV 自动生成 CSV/Markdown/LaTeX 表格
    smoke_test.py           合成数据结构测试
  src/kmfm/
    data.py                 MAT 加载、训练支持域标准化、patch context guard
    splits.py               random-pixel / spatial-block 划分
    model.py                spectral Conv1d、selective SSM、可靠性融合
    engine.py               训练、验证、测试与 artifact 保存
    metrics.py              OA/AA/Kappa 的单一混淆矩阵实现
  tests/
```

## 2. 数据准备

把数据放入 Google Drive，例如：

```text
MyDrive/Colab/Datasets/
  PaviaU.mat
  PaviaU_gt.mat
  Houston_data.mat
  Houston_gt.mat
  KSC.mat
  KSC_gt.mat
  Botswana.mat
  Botswana_gt.mat
```

常见 key：

| 数据集 | HSI key | GT key |
|---|---|---|
| Pavia University | `paviaU` | `paviaU_gt` |
| Houston 2013（附件代码版本） | `hsi` | `groundT` |
| KSC | `KSC` | `KSC_gt` |
| Botswana | `Botswana` | `Botswana_gt` |

如果不填写 key，加载器会选择最大的 3-D 和 2-D 数组，并自动尝试对齐维度。正式实验仍建议显式记录 key。

不要把不同来源、不同标签版本的 Houston 文件混用。每个 split artifact 会保存数据文件 SHA-256。

## 3. Colab 最快运行方式

打开 [KMFM_LASSF_Colab.ipynb](notebooks/KMFM_LASSF_Colab.ipynb)，按顺序执行。Notebook 固定使用：

```text
/content/drive/MyDrive/Colab/Unsupervised/KMFM/
/content/drive/MyDrive/Colab/Datasets/
```

Notebook 会先按 SASM-Mamba 的方式更新 GitHub 仓库，再安装当前代码。

### GitHub → Colab 更新方式

默认仓库地址为 `https://github.com/chenzizi07/KMFM.git`，可在 Notebook 第一 cell 修改 `REPO_URL` 或设置 `KMFM_REPO_URL`。首次使用时，如果目标目录为空，Notebook 会 clone；如果已经是 Git 仓库，会执行：

```bash
cd /content/drive/MyDrive/Colab/Unsupervised/KMFM
bash scripts/colab_update.sh
```

`scripts/colab_git.py` 会尝试读取 Colab Secrets 中的 `GITHUB_TOKEN`、`GH_TOKEN`、`GITHUB_PAT` 以及 SASM-Mamba 常见命名，并通过临时 `GIT_ASKPASS` 使用；token 不会写入 remote URL、Notebook、Drive 或日志。如果仓库公开，则不需要 token。

不要把 token 直接写进 Notebook 或命令行。在 Colab 左侧钥匙图标的 Secrets 面板中创建 `GITHUB_TOKEN`，并允许当前 Notebook 访问。

首次 clone 私有仓库时，建议直接使用 Notebook 的安全 clone cell；shell `colab_install.sh` 适用于仓库已存在或公开仓库。

然后 Notebook 会继续执行：

1. 挂载 Google Drive；
2. 解压/安装本工程；
3. 运行合成数据 smoke test；
4. 修改数据路径与 key；
5. 生成 RP/SB split；
6. 先跑 1 个 seed、30–50 epochs 的 pilot；
7. 检查 run artifact 和 `verify_run.py`；
8. 再运行正式 seeds 和消融。

推荐 Colab GPU：T4/L4/A100。默认实现不依赖 `mamba-ssm` CUDA 扩展，普通 Colab PyTorch 即可运行。

## 4. 命令行示例

安装：

```bash
pip install -r requirements-colab.txt
pip install -e .
python scripts/smoke_test.py
```

生成 random-pixel split：

```bash
python scripts/make_split.py \
  --data-path /content/drive/MyDrive/Colab/Datasets/PaviaU.mat \
  --gt-path /content/drive/MyDrive/Colab/Datasets/PaviaU_gt.mat \
  --data-key paviaU --gt-key paviaU_gt \
  --protocol random_pixel \
  --train-per-class 30 --val-per-class 10 \
  --seed 0 \
  --output /content/drive/MyDrive/Colab/Unsupervised/KMFM/splits/pavia/random_pixel/seed_0.npz
```

生成 spatial-block split：

```bash
python scripts/make_split.py \
  --data-path /content/drive/MyDrive/Colab/Datasets/PaviaU.mat \
  --gt-path /content/drive/MyDrive/Colab/Datasets/PaviaU_gt.mat \
  --data-key paviaU --gt-key paviaU_gt \
  --protocol spatial_block \
  --train-per-class 30 --val-per-class 10 \
  --min-test-per-class 20 \
  --block-size 32 --buffer-pixels 3 --trials 256 \
  --seed 0 \
  --output /content/drive/MyDrive/Colab/Unsupervised/KMFM/splits/pavia/spatial_block/seed_0.npz
```

如果某个类别无法在三个空间区域中都满足样本要求，脚本会明确失败并打印每类可用数。此时应调整 block、buffer、region ratio 或固定样本数，不能静默删除该类别。

训练一个不可追加覆盖的 run：

```bash
python scripts/train.py --config configs/pavia_spatial_block_example.json
```

若相同 `experiment/dataset/protocol/model/seed` 已成功运行，脚本会拒绝覆盖。修改配置中的 `output.experiment` 开始一个新实验批次。

复核结果：

```bash
python scripts/verify_run.py --run-dir /content/drive/MyDrive/Colab/Unsupervised/KMFM/results/pilot_v1/...
```

汇总：

```bash
python scripts/aggregate.py \
  --results-root /content/drive/MyDrive/Colab/Unsupervised/KMFM/results/formal_v1 \
  --output-dir /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/formal_v1 \
  --reference-model lassf_conv1d_concat_h64
```

`summary.csv` 同时给出 0–1 比例和百分数形式的 mean、sample SD、95% Student-t CI；`paired_tests.csv` 给出共享 split hash/seed 的 paired t-test、Wilcoxon、Cohen's dz 和 Holm 校正结果。

从真实汇总结果直接生成论文表格：

```bash
python scripts/export_table.py \
  --summary /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/formal_v1/summary.csv \
  --protocol spatial_block --metric oa \
  --output-prefix /content/drive/MyDrive/Colab/Unsupervised/KMFM/reports/formal_v1/main_sb_oa
```

该命令同时生成 `.csv`、`.md` 和使用 `booktabs` 的 `.tex`，避免向论文手工复制数字。

## 5. 最低实验矩阵

先做四个核心模型，不要一次增加十几个模块：

| 模型名建议 | spectral | fusion | 目的 |
|---|---|---|---|
| `lassf_mlp_concat_h64` | MLP | concat | 参数合理的普通基线 |
| `lassf_conv1d_concat_h64` | Conv1d | concat | 检验正确谱算子 |
| `lassf_conv1d_gate_h64` | Conv1d | plain gate | 检验普通动态融合 |
| `lassf_conv1d_reliability_h64` | Conv1d | reliability | 主模型 |
| `lassf_conv1d_spatial_only_h64` | Conv1d | spatial only | 正式实验的空间分支消融 |
| `lassf_conv1d_spectral_only_h64` | Conv1d | spectral only | 正式实验的光谱分支消融 |

每个模型使用相同 RP/SB mask 和 seeds。建议流程：

- pilot：PaviaU + KSC，2 seeds，50 epochs；
- go/no-go：主模型在 SB 上有信号后再扩展；
- 正式：四数据集，至少 5 个 SB seeds、10 个 RP seeds；
- 结果稳定且 Colab 配额允许时，再把 SB 扩展到 10 seeds。

## 6. 结果目录

```text
results/{experiment}/{dataset}/{protocol}/{model}/seed_{seed}/
  status.json
  resolved_config.json
  data_manifest.json
  environment.json
  checkpoint_best.pt
  curves.csv
  prediction.npy
  ground_truth_eval.npy
  confusion_matrix.npy
  confusion_matrix.csv
  gate.npy
  spatial_entropy.npy
  spectral_entropy.npy
  metrics.json
  manifest.json
```

同一 run 目录不会追加历史结果。`manifest.json` 保存输入和输出文件哈希。

## 7. 论文写作边界

- RP 结果只能表述为 random-pixel protocol performance；
- SB 才用于跨区域/空间互斥主张；
- pure-PyTorch selective SSM 不能写成官方 Mamba；
- 如果 spectral Conv1d 不优于 MLP，就不声称它是贡献；
- 如果 reliability gate 不优于 concat/plain gate，就不能作为主创新；
- 所有论文表格必须来自聚合脚本，不手工改数。

论文重构建议见 [PAPER_REBUILD_FROM_RESULTS.md](PAPER_REBUILD_FROM_RESULTS.md)。
