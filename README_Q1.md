# 问题1运行说明

问题1从附件1的100条原始视频重新提取三种模态特征，生成独立的 `768/768/768` 接口。附件2的 `768/74/35` 特征仅供问题2和问题3使用，不能混用。

## 远程环境准备

在服务器上执行：

```bash
source /usr/local/iCompute/etc/profile.d/conda.sh
conda activate eeeeee
conda install -y -c conda-forge ffmpeg=7.1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
python -m pip install -r requirements-q1.txt
```

安装后记录：

```bash
python -m pip freeze > runs/environment-pip-freeze.txt
ffmpeg -version | head -n 1
nvidia-smi
```

## 运行

代码和配置同步到 `/home/user/EEEEEE` 后执行：

```bash
source /usr/local/iCompute/etc/profile.d/conda.sh
conda activate eeeeee
cd /home/user/EEEEEE
tmux new -s eeeeee-q1
bash scripts/run_q1.sh --run-id 20260923_q1_001 --config configs/q1.yaml
```

试运行可以使用：

```bash
bash scripts/run_q1.sh --run-id 20260923_q1_smoke --limit 1 --allow-failures --skip-model-hash
```

运行完成后，只有出现 `runs/<run_id>/DONE` 才下载结果。结果包括 `outputs/q1_submission/` 和 `outputs/q1_audit/`。

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
bash scripts/validate_q1.sh \
  --artifact runs/20260923_q1_001/outputs/q1_submission/q1_aligned_50.pkl \
  --manifest runs/20260923_q1_001/outputs/q1_submission/manifest.csv \
  --expected-count 100
```

压缩包不包含模型权重、缓存、未对齐审计文件或身份信息。提交前检查 `package_size.json` 中的 `within_limit`。

