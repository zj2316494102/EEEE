from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _finite_stat(values: np.ndarray, quantile: float | None = None) -> float | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    if quantile is None:
        return float(array.mean())
    return float(np.quantile(array, quantile))


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "mean": _finite_stat(array),
        "median": _finite_stat(array, 0.50),
        "p10": _finite_stat(array, 0.10),
        "p90": _finite_stat(array, 0.90),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def _unit_interval(values: np.ndarray, valid: np.ndarray, *, low: float = 0.10, high: float = 0.90) -> np.ndarray:
    """Robustly map valid values to [0, 1] without turning invalid bins on."""

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(array)
    output = np.zeros(array.shape, dtype=np.float32)
    if not valid_mask.any():
        return output
    reference = array[valid_mask]
    lower = float(np.quantile(reference, low))
    upper = float(np.quantile(reference, high))
    if upper <= lower + 1e-8:
        output[valid_mask] = 1.0
        return output
    output[valid_mask] = np.clip((reference - lower) / (upper - lower), 0.0, 1.0)
    return output


def compute_trimodal_temporal_alignment(
    text_mask: np.ndarray,
    text_confidence: np.ndarray,
    audio_mask: np.ndarray,
    audio_prosody: np.ndarray,
    audio_prosody_mask: np.ndarray,
    vision_mask: np.ndarray,
    vision_quality_aligned: np.ndarray,
    *,
    epsilon: float = 1e-8,
    audio_rms_quantile_low: float = 0.10,
    audio_rms_quantile_high: float = 0.90,
    min_joint_bins: int = 3,
) -> dict[str, Any]:
    """Compute the single Question 1 trimodal temporal alignment score.

    The three encoders do not share a semantic feature space, so this function
    never takes a cosine between text/audio/vision embeddings.  It compares
    confidence-weighted temporal support on the common 50-bin presentation
    timeline.  The score is a fuzzy three-way Jaccard index:

        sum(min(text_support, audio_support, vision_support)) /
        sum(max(text_support, audio_support, vision_support))

    ``text_confidence`` downweights ASR fallback timing, ``audio_prosody``
    supplies speech activity from RMS and voiced status, and
    ``vision_quality_aligned`` supplies face observation quality.
    """

    text_valid = np.asarray(text_mask, dtype=bool).reshape(-1)
    text_quality = np.asarray(text_confidence, dtype=np.float32).reshape(-1)
    audio_valid = np.asarray(audio_mask, dtype=bool).reshape(-1)
    prosody = np.asarray(audio_prosody, dtype=np.float32)
    prosody_valid = np.asarray(audio_prosody_mask, dtype=bool).reshape(-1)
    vision_valid = np.asarray(vision_mask, dtype=bool).reshape(-1)
    vision_quality = np.asarray(vision_quality_aligned, dtype=np.float32).reshape(-1)
    if prosody.ndim != 2 or prosody.shape[1] < 3:
        raise ValueError(f"audio_prosody must have shape [T,>=3], got {prosody.shape}")
    length = text_valid.size
    if any(
        array.size != length
        for array in (text_quality, audio_valid, prosody_valid, vision_valid, vision_quality)
    ) or prosody.shape[0] != length:
        raise ValueError(
            "Trimodal support arrays must share the aligned length: "
            f"text={text_valid.shape}, confidence={text_quality.shape}, "
            f"audio={audio_valid.shape}, prosody={prosody.shape}, "
            f"vision={vision_valid.shape}, vision_quality={vision_quality.shape}"
        )

    text_support = np.clip(text_quality, 0.0, 1.0) * text_valid.astype(np.float32)
    text_support[~np.isfinite(text_support)] = 0.0

    prosody_finite = np.isfinite(prosody[:, 1]) & np.isfinite(prosody[:, 2])
    prosody_joint = audio_valid & prosody_valid & prosody_finite
    if not 0.0 <= audio_rms_quantile_low < audio_rms_quantile_high <= 1.0:
        raise ValueError(
            "audio RMS quantiles must satisfy 0 <= low < high <= 1: "
            f"low={audio_rms_quantile_low}, high={audio_rms_quantile_high}"
        )
    rms_activity = _unit_interval(
        np.maximum(prosody[:, 1], 0.0),
        prosody_joint,
        low=audio_rms_quantile_low,
        high=audio_rms_quantile_high,
    )
    voiced_activity = np.clip(np.nan_to_num(prosody[:, 2], nan=0.0), 0.0, 1.0)
    audio_activity = 0.5 * rms_activity + 0.5 * voiced_activity
    audio_support = audio_activity * prosody_joint.astype(np.float32)

    vision_support = np.clip(vision_quality, 0.0, 1.0) * vision_valid.astype(np.float32)
    vision_support[~np.isfinite(vision_support)] = 0.0

    supports = np.stack([text_support, audio_support, vision_support], axis=0)
    intersection = np.min(supports, axis=0)
    union = np.max(supports, axis=0)
    joint_support_bins = int(np.count_nonzero(intersection > epsilon))
    union_support_bins = int(np.count_nonzero(union > epsilon))
    if union_support_bins == 0:
        return {
            "trimodal_temporal_alignment_score": None,
            "joint_support_bins": 0,
            "alignment_status": "no_modality_support",
        }

    score = float(np.clip(intersection.sum() / (union.sum() + epsilon), 0.0, 1.0))
    return {
        "trimodal_temporal_alignment_score": score,
        "joint_support_bins": joint_support_bins,
        "alignment_status": "ok" if joint_support_bins >= min_joint_bins else "insufficient_joint_support",
    }


def aggregate_alignment_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the single primary trimodal temporal alignment metric."""

    values = np.asarray(
        [record.get("trimodal_temporal_alignment_score", np.nan) for record in records],
        dtype=np.float64,
    )
    joint_support_bins = np.asarray(
        [record.get("joint_support_bins", np.nan) for record in records],
        dtype=np.float64,
    )
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("alignment_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "trimodal_temporal_alignment_score": _summary(values),
        "joint_support_bins": _summary(joint_support_bins),
        "alignment_status_counts": status_counts,
    }


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    cjk_font_paths = [
        Path("/home/user/.trae-server/bin/stable-576b4799102d1c442511b50905e566dfd190113d-debian10/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf"),
        Path("/home/user/.vscode-server/extensions/ai-completion/resource/aiserver/resources/font/HeiTi.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    cjk_font_name = None
    for cjk_font_path in cjk_font_paths:
        if cjk_font_path.exists():
            try:
                font_manager.fontManager.addfont(str(cjk_font_path))
                cjk_font_name = font_manager.FontProperties(fname=str(cjk_font_path)).get_name()
                break
            except Exception:
                continue
    plt.rcParams["font.family"] = [cjk_font_name, "DejaVu Sans"] if cjk_font_name else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    fig.clear()
    import matplotlib.pyplot as plt

    plt.close(fig)


def _mean_or_zero(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def render_alignment_summary_table(
    output_path: Path,
    effective_rows: list[dict[str, Any]],
    alignment_summary: dict[str, Any],
    feature_dims: dict[str, int],
    quality_records: list[dict[str, Any]],
) -> None:
    plt = _matplotlib()
    metric_summary = alignment_summary.get("trimodal_temporal_alignment_score", {})
    metric_mean = metric_summary.get("mean")
    metric_median = metric_summary.get("median")
    metric_count = int(metric_summary.get("count") or 0)
    record_values: list[tuple[float, str]] = []
    for record in quality_records:
        value = record.get("trimodal_temporal_alignment_score")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            record_values.append((value, str(record.get("id", ""))))
    record_values.sort(key=lambda item: item[0])
    status_colors = {"ok": "#3a9b57", "degraded": "#e49a2f", "missing": "#b8b8b8"}
    status_labels = {"ok": "正常", "degraded": "降级", "missing": "缺失"}
    alignment_status_counts = {
        str(key): int(value)
        for key, value in dict(alignment_summary.get("alignment_status_counts", {})).items()
    }
    quality_status_counts = {
        status: sum(1 for row in effective_rows if str(row.get("quality_status", "")) == status)
        for status in ("ok", "degraded", "missing")
    }

    text_words = _mean_or_zero(effective_rows, "text_reliable_word_count")
    text_total_words = _mean_or_zero(effective_rows, "text_raw_words")
    audio_bins = _mean_or_zero(effective_rows, "audio_aligned_valid_bins")
    vision_bins = _mean_or_zero(effective_rows, "vision_aligned_valid_bins")
    length_values = np.asarray([text_words, audio_bins, vision_bins], dtype=np.float64)
    length_denominators = np.asarray([max(text_total_words, 1.0), 50.0, 50.0], dtype=np.float64)
    length_ratios = np.clip(length_values / length_denominators, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    fig.suptitle("问题1：特征、有效长度与三模态对齐质量概览", fontsize=16)

    # Panel 1: effective length. The denominator is the average raw word/frame count
    # for each modality, while the labels retain the original units.
    length_axis = axes[0]
    names = ["文本", "音频", "视觉"]
    bars = length_axis.bar(names, length_ratios * 100.0, color=["#7565ac", "#ef7e25", "#3a9b57"])
    length_axis.set_title("平均有效长度")
    length_axis.set_ylabel("有效比例（%）")
    length_axis.set_ylim(0.0, 110.0)
    length_axis.grid(axis="y", alpha=0.25)
    length_labels = [f"{text_words:.1f}/{text_total_words:.1f} 词", f"{audio_bins:.1f}/50 段", f"{vision_bins:.1f}/50 段"]
    for bar, ratio, label in zip(bars, length_ratios, length_labels):
        length_axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 2.0, f"{ratio * 100.0:.1f}%\n{label}", ha="center", va="bottom", fontsize=9)

    # Panel 2: the one primary trimodal alignment metric.
    metric_axis = axes[1]
    metric_names = ["均值", "中位数"]
    metric_values = np.asarray([metric_mean or 0.0, metric_median or 0.0], dtype=np.float64)
    metric_bars = metric_axis.bar(metric_names, metric_values, color=["#5b8db8", "#806db2"], width=0.56)
    metric_axis.set_title("三模态联合时间对齐分数（TMA）")
    metric_axis.set_ylabel("分数")
    metric_axis.set_ylim(0.0, max(0.20, float(metric_values.max()) * 1.35))
    metric_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(metric_bars, metric_values):
        metric_axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    metric_axis.text(0.98, 0.95, f"可计算样本：{metric_count}/{len(quality_records)}", transform=metric_axis.transAxes, ha="right", va="top", fontsize=9)

    # Panel 3: sample-level quality status and the low-support count.
    status_axis = axes[2]
    status_names = [status_labels[name] for name in ("ok", "degraded", "missing")]
    status_values = [quality_status_counts[name] for name in ("ok", "degraded", "missing")]
    status_bars = status_axis.bar(status_names, status_values, color=[status_colors[name] for name in ("ok", "degraded", "missing")])
    status_axis.set_title("样本质量状态")
    status_axis.set_ylabel("样本数")
    status_axis.set_ylim(0.0, max(10.0, max(status_values) * 1.18))
    status_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(status_bars, status_values):
        status_axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(value), ha="center", va="bottom", fontsize=10)
    status_axis.text(0.98, 0.95, f"共同支持≥3段：{alignment_status_counts.get('ok', 0)}/{len(quality_records)}", transform=status_axis.transAxes, ha="right", va="top", fontsize=9)

    _save_figure(fig, output_path)


def render_label_feature_overview(
    output_path: Path,
    labels: np.ndarray,
    feature_dims: dict[str, int],
) -> None:
    plt = _matplotlib()
    labels = np.asarray(labels, dtype=np.float64)
    labels = labels[np.isfinite(labels)]
    class_counts = [int((labels < 0).sum()), int((labels == 0).sum()), int((labels > 0).sum())]
    class_names = ["Negative", "Neutral", "Positive"]
    colors = ["#2f7fb8", "#37a657", "#ef6411"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    bars = axes[0].bar(class_names, class_counts, color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_title("附件1标签类别")
    axes[0].set_ylabel("样本数")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, class_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=10)

    if labels.size:
        unique = np.unique(labels)
        bins = np.linspace(unique[0] - 0.5, unique[0] + 0.5, 4) if unique.size == 1 else min(max(int(unique.size * 2), 6), 20)
        axes[1].hist(labels, bins=bins, color="#4f7cac", edgecolor="white")
    axes[1].set_title("数值标签分布")
    axes[1].set_xlabel("标签值")
    axes[1].set_ylabel("样本数")
    axes[1].grid(axis="y", alpha=0.25)

    names = list(feature_dims.keys())
    dims = [feature_dims[name] for name in names]
    bars = axes[2].bar(names, dims, color=["#7565ac", "#ef7e25", "#3a9b57"], edgecolor="#333333", linewidth=0.5)
    axes[2].set_title("问题1主特征维度")
    axes[2].set_ylabel("维度")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, dim in zip(bars, dims):
        axes[2].text(bar.get_x() + bar.get_width() / 2, dim, str(dim), ha="center", va="bottom", fontsize=10)
    fig.suptitle("问题1：特征维度与标签概览", fontsize=14)
    _save_figure(fig, output_path)


def _word_position_matrix(raw_audits: list[dict[str, Any]], grids: np.ndarray, bins: int = 50) -> np.ndarray:
    matrix = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)
    for audit, grid in zip(raw_audits, grids):
        words = (audit.get("text", {}) or {}).get("words", []) or []
        if not words:
            continue
        for word_index, word in enumerate(words):
            start, end = word.get("start"), word.get("end")
            if start is None or end is None or float(end) <= float(start):
                continue
            center = (float(start) + float(end)) / 2.0
            time_index = int(np.searchsorted(grid[:, 1], center, side="right"))
            time_index = int(np.clip(time_index, 0, bins - 1))
            word_position = int(round((bins - 1) * word_index / max(len(words) - 1, 1)))
            confidence = float(np.clip(word.get("confidence", 0.0), 0.0, 1.0))
            if word.get("fallback"):
                confidence *= 0.35
            matrix[word_position, time_index] += confidence
            counts[word_position, time_index] += 1.0
    return np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)


def render_text_alignment_heatmap(
    output_path: Path,
    raw_audits: list[dict[str, Any]],
    grids: np.ndarray,
    text_stack: np.ndarray,
    audio_stack: np.ndarray,
    vision_stack: np.ndarray,
    masks: dict[str, np.ndarray],
    selected_index: int,
) -> None:
    plt = _matplotlib()
    selected_index = int(np.clip(selected_index, 0, len(grids) - 1))
    grid = np.asarray(grids[selected_index], dtype=np.float64)
    matrix = _word_position_matrix([raw_audits[selected_index]], [grid])
    fig, heat_axis = plt.subplots(figsize=(7.8, 6.4), constrained_layout=True)
    image = heat_axis.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
    heat_axis.set_title("典型样本文本词位与时间段对应关系")
    heat_axis.set_xlabel("50段时间位置")
    heat_axis.set_ylabel("归一化文本词位置")
    heat_axis.set_xticks([0, 10, 20, 30, 40, 49])
    heat_axis.set_yticks([0, 10, 20, 30, 40, 49])
    fig.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04, label="文本覆盖权重")
    _save_figure(fig, output_path)


def render_typical_trimodal_features(
    output_path: Path,
    text_stack: np.ndarray,
    audio_stack: np.ndarray,
    vision_stack: np.ndarray,
    masks: dict[str, np.ndarray],
    selected_index: int,
) -> None:
    """Render three temporal feature responses for one representative sample."""

    plt = _matplotlib()
    selected_index = int(np.clip(selected_index, 0, len(text_stack) - 1))
    names = ["文本", "音频", "视觉"]
    values_list = [
        np.asarray(text_stack[selected_index], dtype=np.float32),
        np.asarray(audio_stack[selected_index], dtype=np.float32),
        np.asarray(vision_stack[selected_index], dtype=np.float32),
    ]
    mask_keys = ["text", "audio", "vision"]
    colors = ["#7565ac", "#2f7fb8", "#3a9b57"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)
    for axis, name, values, mask_key, color in zip(axes, names, values_list, mask_keys, colors):
        response = np.linalg.norm(values, axis=1)
        scale = max(float(np.max(response)), 1e-8)
        response = response / scale
        raw_mask = np.asarray(masks.get(mask_key, np.zeros(response.shape[0], dtype=np.uint8)))
        valid = raw_mask[selected_index] if raw_mask.ndim == 2 else raw_mask
        valid = np.asarray(valid, dtype=bool)
        bar_colors = [color if flag else "#d9d9d9" for flag in valid]
        bars = axis.bar(np.arange(response.size), response, color=bar_colors, width=0.82, linewidth=0.25, edgecolor="white")
        for bar, is_valid in zip(bars, valid):
            if not is_valid:
                bar.set_hatch("//")
                bar.set_edgecolor("#999999")
        axis.set_title(f"{name}特征响应")
        axis.set_xlabel("时间段")
        axis.set_ylim(0.0, 1.08)
        axis.set_xticks([0, 10, 20, 30, 40, 49])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("归一化特征范数")
    fig.suptitle("典型样本三模态时序特征", fontsize=14)
    _save_figure(fig, output_path)


def render_alignment_quality_coverage(
    output_path: Path,
    quality_records: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
) -> None:
    plt = _matplotlib()
    values = np.asarray(
        [row.get("trimodal_temporal_alignment_score", np.nan) for row in quality_records],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    if values.size:
        axes[0].hist(values, bins=bins, alpha=0.78, color="#6b5ca5", edgecolor="white")
        axes[0].axvline(float(values.mean()), color="#d14b3f", linestyle="--", label=f"mean={values.mean():.4f}")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "无可计算样本", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("三模态联合时间对齐分数分布")
    axes[0].set_xlabel("三模态联合时间对齐分数（TMA）")
    axes[0].set_ylabel("样本数")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(alpha=0.22)

    names = ["text", "audio", "vision"]
    labels = ["文本", "音频", "视觉"]
    raw_ratios = [_mean_or_zero(effective_rows, f"{name}_raw_valid_ratio") for name in names]
    aligned_ratios = [_mean_or_zero(effective_rows, f"{name}_aligned_valid_ratio") for name in names]
    x = np.arange(3)
    width = 0.36
    bars1 = axes[1].bar(x - width / 2, np.asarray(raw_ratios) * 100.0, width, label="原始/可靠覆盖率", color="#5b8db8")
    bars2 = axes[1].bar(x + width / 2, np.asarray(aligned_ratios) * 100.0, width, label="50段有效覆盖率", color="#7a68ad")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("有效覆盖率（%）")
    axes[1].set_title("三模态有效覆盖率")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    for bars in (bars1, bars2):
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("问题1：三模态对齐分数与有效覆盖率", fontsize=14)
    _save_figure(fig, output_path)


def render_storage_spec(output_path: Path) -> None:
    plt = _matplotlib()
    rows = [
        ["id", "[N]", "string", "video_id + separator + clip_id"],
        ["raw_text", "[N]", "string", "附件1原始文本"],
        ["text", "[N,50,768]", "float16", "文本主特征"],
        ["audio", "[N,50,768]", "float16", "WavLM音频主特征"],
        ["vision", "[N,50,768]", "float16", "人脸/图像主特征"],
        ["masks", "4 x [N,50]", "uint8", "文本/音频/视觉 + 可靠文本"],
        ["text-confidence", "[N,50]", "float32", "文本质量证据"],
        ["vision-quality-aligned", "[N,50]", "float32", "对齐时间段的人脸质量"],
        ["vision-weight-sum", "[N,50]", "float32", "视觉池化支持量"],
        ["audio-prosody", "[N,50,5]", "float32", "语音活动支持计算"],
        ["time-grid", "[N,50,2]", "float32", "展示时间区间"],
        ["alignment-quality.csv", "[N]", "CSV", "唯一三模态联合时间对齐指标"],
        ["raw-unaligned-arrays", "outside package", "-", "仅用于审计，不进入最终提交包"],
    ]
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["字段名（英文）", "形状", "类型/存储", "用途"],
        loc="center",
        cellLoc="left",
        colWidths=[0.23, 0.18, 0.18, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#e8edf5")
            cell.set_text_props(weight="bold")
    ax.set_title("问题1紧凑特征文件存储规范", pad=18, fontsize=14)
    ax.text(0.5, -0.035, "最终提交包仅包含紧凑的对齐特征与审计摘要；未对齐原始数组保留在提交包之外。", ha="center", transform=ax.transAxes, fontsize=9)
    _save_figure(fig, output_path)


def render_quality_reports(
    output_dir: Path,
    *,
    manifest: list[dict[str, Any]],
    raw_audits: list[dict[str, Any]],
    grids: np.ndarray,
    text_stack: np.ndarray,
    audio_stack: np.ndarray,
    vision_stack: np.ndarray,
    masks: dict[str, np.ndarray],
    effective_rows: list[dict[str, Any]],
    quality_records: list[dict[str, Any]],
    labels: np.ndarray,
    selected_index: int,
    feature_dims: dict[str, int],
) -> dict[str, str]:
    """Render the five compact reports from actual Question 1 outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_alignment_quality(quality_records)
    paths = {
        "alignment_summary_table": output_dir / "图1_三模态时序对齐质量概览.png",
        "label_feature_overview": output_dir / "图2_标签与特征维度概览.png",
        "text_alignment_heatmap": output_dir / "图3a_典型样本文本位置热力图.png",
        "typical_trimodal_features": output_dir / "图3b_典型样本三模态特征响应.png",
        "alignment_quality_coverage": output_dir / "图4_三模态对齐分数与有效覆盖率.png",
        "storage_spec": output_dir / "图5_特征文件存储规范.png",
    }
    render_alignment_summary_table(paths["alignment_summary_table"], effective_rows, summary, feature_dims, quality_records)
    render_label_feature_overview(paths["label_feature_overview"], labels, feature_dims)
    render_text_alignment_heatmap(
        paths["text_alignment_heatmap"],
        raw_audits,
        grids,
        text_stack,
        audio_stack,
        vision_stack,
        masks,
        selected_index,
    )
    render_typical_trimodal_features(
        paths["typical_trimodal_features"],
        text_stack,
        audio_stack,
        vision_stack,
        masks,
        selected_index,
    )
    render_alignment_quality_coverage(paths["alignment_quality_coverage"], quality_records, effective_rows)
    render_storage_spec(paths["storage_spec"])
    return {key: str(path) for key, path in paths.items()}
