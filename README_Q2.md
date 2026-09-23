# 问题2运行说明

问题2使用附件二 `aligned_50.pkl` 的 `text/audio/vision` 连续特征（`768/74/35`），训练时只在训练集在线模拟连续时间块缺失，验证集使用固定缺失矩阵，附件三只在配置冻结后推理。

## 本地或远程运行

先安装远程环境依赖：

```bash
conda activate eeeeee
python -m pip install -r requirements-q2.txt
```

同步代码后，在远程服务器执行：

```bash
bash tools/run_q2.sh --run-id 20260923_q2_001 --config configs/q2.yaml
```

长任务建议放在 `tmux` 中。运行目录为 `runs/<run_id>/`，启动时创建 `RUNNING`，全部输出写完后才创建 `DONE`；异常时创建 `FAILED`。只有 `DONE` 存在时才执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\pull_results_from_remote.ps1 -RunId 20260923_q2_001
```

快速检查：

```bash
python -m q2.pipeline --config configs/q2.yaml --run-id q2_smoke \
  --limit 8 --smoke --compact-validation --skip-input-hash
```

## 输出

`runs/<run_id>/outputs/问题2/` 包含：

- `model/robust_student.pt`：最终选中的轻量学生模型；
- `model/normalization.npz`：只由训练集有效位置拟合的标准化统计量；
- `validation/metrics_by_scenario.csv`、`confusion_matrix.csv`、`mask_scenarios.csv`、`training_history.csv`、`ablation_results.csv`；
- `q2_attachment3_predictions.csv`：附件三预测主文件；
- `q2_attachment3_audit.csv`、`inference_manifest.json`：ID、覆盖率、非有限值和模型校验审计；
- `run_manifest.json`、`README.md`：数据哈希、掩码规则、配置和运行摘要。

训练恢复文件位于 `runs/<run_id>/checkpoints/`，最终推理模型仍位于
`outputs/问题2/model/robust_student.pt`。

附件三对齐文件当前提供 `text_bert` 整数输入而非 `text` 连续特征。程序不会把它当作连续文本通道，而是记录 `text_bert_ignored_count` 并将文本模态标记为不可用；如后续提供对应的 `text [N,50,768]`，放入同一接口即可自动使用。

## 中断后继续运行

每个运行实例都会在 `runs/<run_id>/checkpoints/` 下保存可恢复检查点。
检查点只在一个 epoch 及其验证完成后通过原子替换写入，包含模型参数、优化器状态、最佳模型、训练历史以及随机数状态。

使用相同配置恢复中断的运行：

```bash
bash tools/run_q2.sh --resume 20260923_q2_001 --config configs/q2.yaml
```

`--resume` 也可以接收完整的运行目录。已完成的训练阶段会直接加载检查点，未完成阶段从最近一个完整 epoch 继续。已经存在 `DONE` 的运行不能恢复；如需重新实验，请使用新的 `--run-id`。
