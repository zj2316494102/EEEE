# 问题1运行说明

问题1从附件1的100条原始视频重新提取三种模态特征，生成独立的 `768/768/768` 接口。附件2的 `768/74/35` 特征仅供问题2和问题3使用，不能混用。

## 远程环境准备

在服务器上执行：

```bash
source /usr/local/iCompute/etc/profile.d/conda.sh
conda activate eeeeee
conda install -y -c conda-forge ffmpeg=7.1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
python -m pip install -r requirements.txt
```

安装后记录（正式运行会自动写入每个 run 的 `environment/`）：

```bash
python -m pip freeze > results/问题1/environment-pip-freeze.txt
ffmpeg -version | head -n 1
nvidia-smi
```

## 运行

代码和配置同步到 `/home/user/EEEEEE` 后执行。运行结果统一写入 `results/问题1/<run_id>/`：

```bash
source /usr/local/iCompute/etc/profile.d/conda.sh
conda activate eeeeee
cd /home/user/EEEEEE
tmux new -s eeeeee-q1
python -m q1.pipeline --run-id 20260923_q1_001 --config configs/q1.yaml
```

试运行可以使用：

```bash
python -m q1.pipeline --run-id 20260923_q1_smoke --limit 1 --allow-failures --skip-model-hash
```

运行完成后，只有出现 `results/问题1/<run_id>/DONE` 才下载结果。结果包括 `outputs/q1_submission/`、`outputs/q1_audit/` 和包含代码/环境/典型审计材料的 `outputs/q1_submission_<run_id>.zip`。

## 输出接口

`q1_aligned_50.pkl` 包含：

```text
text   [N, 50, 768] float16
audio  [N, 50, 768] float16
vision [N, 50, 768] float16
```

每个模态同时保存有效掩码、有效长度和每条样本的50段时间网格。未检测到人脸、无声或文本低置信度区间不会被静默删除。

## 验证

```bash
python -m q1.validate \
  --artifact results/问题1/20260923_q1_001/outputs/q1_submission/q1_aligned_50.pkl \
  --manifest results/问题1/20260923_q1_001/outputs/q1_submission/manifest.csv \
  --raw-audit results/问题1/20260923_q1_001/outputs/q1_audit/q1_unaligned.pkl \
  --alignment-log results/问题1/20260923_q1_001/outputs/q1_submission/alignment_log.csv \
  --text-review results/问题1/20260923_q1_001/outputs/q1_submission/text_review.csv \
  --duration-audit results/问题1/20260923_q1_001/outputs/q1_submission/duration_audit.csv \
  --package-zip results/问题1/20260923_q1_001/outputs/q1_submission_20260923_q1_001.zip \
  --expected-count 100
```

压缩包包含问题1核心代码、配置、环境快照、`tools/`运行入口、典型样本图/侧车和提交结果；不包含模型权重、缓存、源视频或完整未对齐审计文件。提交前检查 `package_size.json` 中的 `within_limit`。
