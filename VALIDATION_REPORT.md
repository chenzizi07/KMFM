# Validation Report

日期：2026-07-23

## 已执行验证

1. `python -m compileall`：`src/`、`scripts/`、`tests/` 全部通过。
2. Colab notebook JSON：14 个 cell，nbformat 4，结构有效。
3. `scripts/smoke_test.py`：通过。
4. `pytest -q`：5 passed。
5. random-pixel 合成 MAT 端到端集成：成功完成加载、标准化、2 epochs 训练、最佳 checkpoint 恢复、测试和 artifact manifest。
6. spatial-block 合成 MAT 端到端集成：成功完成空间块划分、context guard、训练和测试。
7. 聚合器合成多 seed 测试：成功生成分组汇总、95% CI 输入字段和配对检验表。
8. 表格导出测试：成功从汇总 CSV 生成 CSV、Markdown 和 LaTeX `booktabs` 表格。
9. CLI：`make_split.py --help`、`train.py --help`、`aggregate.py --help` 均正常。

## 测试环境

- Windows 本地隔离环境：`.venv-test`
- Python：3.13
- PyTorch：2.13.0+cpu
- 正式 Colab 设计：使用 Colab 自带 CUDA PyTorch，不依赖 `mamba-ssm` 编译扩展。

## 尚未执行

当前工作区没有 Pavia University、Houston 2013、KSC 和 Botswana 的原始 MAT 数据，因此没有运行真实数据实验，也没有生成任何论文结果数字。真实实验必须在 Colab 挂载数据后按 notebook 执行。

## 2026-07-23 Colab/GitHub 适配复验

- 固定项目目录：`/content/drive/MyDrive/Colab/Unsupervised/KMFM/`。
- 固定数据目录：`/content/drive/MyDrive/Colab/Datasets/`，Notebook 已登记 Pavia University、Houston 2013、KSC、Botswana 四个数据集。
- GitHub 更新默认使用 `https://github.com/chenzizi07/KMFM.git`，并兼容 SASM-Mamba 已有的 Colab Secret 名称。
- 私有仓库认证使用权限为 `0700` 的临时 `GIT_ASKPASS` shell 脚本；令牌不写入 remote URL、Notebook、Drive 或日志。
- 修订后复验：`compileall` 通过，Notebook JSON 及 14 个 cell 有效，`pytest` 为 5 passed，synthetic smoke test 通过，CLI help 正常。

## 科学完整性边界

- 合成数据只验证代码结构，不代表方法性能；
- SCI 四区发表取决于真实空间互斥实验是否支持主张；
- 任何论文数值必须通过 `scripts/verify_run.py` 从 prediction/ground truth 重新验证。
