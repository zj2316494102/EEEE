from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _safe_title(value: Any) -> str:
    return str(value).replace("$", r"\$")


def _load_waveform(video_path: Path, duration: float, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    from .utils import extract_audio

    audio = extract_audio(video_path, sample_rate)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    times = np.arange(audio.size, dtype=np.float64) / float(sample_rate)
    keep = times <= float(duration) + 1e-6
    times = times[keep]
    audio = audio[keep]
    max_points = 5000
    if audio.size > max_points:
        indices = np.linspace(0, audio.size - 1, max_points, dtype=np.int64)
        times, audio = times[indices], audio[indices]
    scale = max(float(np.max(np.abs(audio))), 1e-6)
    return times, (audio / scale).astype(np.float32)


def _load_keyframes(
    video_path: Path,
    frame_indices: np.ndarray,
    centers: np.ndarray,
    face_bbox: np.ndarray,
    duration: float,
    max_frames: int = 6,
) -> list[tuple[float, np.ndarray, int]]:
    import cv2

    if len(frame_indices) == 0:
        return []
    choose = np.linspace(0, len(frame_indices) - 1, min(max_frames, len(frame_indices)), dtype=np.int64)
    cap = cv2.VideoCapture(str(video_path))
    output: list[tuple[float, np.ndarray, int]] = []
    for position in choose:
        frame_index = int(frame_indices[position])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        bbox = np.asarray(face_bbox[position], dtype=np.float32) if len(face_bbox) > position else np.zeros(4)
        if bbox.shape == (4,) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            left, top = int(np.clip(bbox[0], 0, 1) * width), int(np.clip(bbox[1], 0, 1) * height)
            right, bottom = int(np.clip(bbox[2], 0, 1) * width), int(np.clip(bbox[3], 0, 1) * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), max(1, width // 320))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        center = float(centers[position]) if len(centers) > position else float(frame_index) / 25.0
        if 0.0 <= center <= float(duration):
            output.append((center, rgb, frame_index))
    cap.release()
    return output


def render_sample_figure(
    sample_id: str,
    aligned: dict[str, Any],
    output_path: Path,
    raw_audit: dict[str, Any] | None = None,
    source_path: Path | None = None,
    duration_container: float | None = None,
    duration_alignment: float | None = None,
) -> None:
    """Render a same-time-axis audit figure with text, audio and video evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(aligned["time_grid"], dtype=np.float64)
    centers = grid.mean(axis=1)
    duration = float(duration_alignment if duration_alignment is not None else grid[-1, 1])
    masks = {key: np.asarray(value) for key, value in aligned["masks"].items()}
    text = np.asarray(aligned["text"], dtype=np.float32)
    audio = np.asarray(aligned["audio"], dtype=np.float32)
    vision = np.asarray(aligned["vision"], dtype=np.float32)
    emotion = np.asarray(aligned["vision_emotion_probs"], dtype=np.float32)
    text_confidence = np.asarray(aligned.get("text_confidence", np.zeros(50)), dtype=np.float32)
    text_low = np.asarray(aligned.get("text_low_confidence_mask", np.zeros(50)), dtype=np.float32)
    raw_audit = raw_audit or {}
    raw_text = raw_audit.get("text", {}) or {}
    raw_audio = raw_audit.get("audio", {}) or {}
    raw_vision = raw_audit.get("vision", {}) or {}

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 14),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.8, 3.0, 1.5, 1.5]},
    )
    title_duration = f"container={duration_container:.3f}s, alignment={duration:.3f}s" if duration_container else f"alignment={duration:.3f}s"
    fig.suptitle(f"Question 1 same-axis multimodal audit: {_safe_title(sample_id)}\n{title_duration}")

    text_axis = axes[0]
    words = raw_text.get("words", []) or []
    for word in words:
        start, end = word.get("start"), word.get("end")
        if start is None or end is None or end <= start:
            continue
        text_axis.axvspan(float(start), float(end), color="tab:blue", alpha=0.10)
        text_axis.text(
            (float(start) + float(end)) / 2.0,
            0.52,
            str(word.get("word", "")),
            ha="center",
            va="center",
            rotation=45,
            fontsize=7,
            color="tab:red" if word.get("fallback") else "tab:blue",
        )
    text_axis.set_ylim(0, 1)
    text_axis.set_yticks([0.52], ["words\nred=fallback"])
    text_axis.set_title("Provided transcript words and persisted word-level timestamps")

    audio_axis = axes[1]
    if source_path and source_path.exists():
        try:
            waveform_time, waveform = _load_waveform(source_path, duration)
            if len(waveform):
                audio_axis.plot(waveform_time, waveform, color="tab:orange", linewidth=0.45, label="waveform")
        except Exception as exc:  # pragma: no cover - audit visualization fallback
            audio_axis.text(0.01, 0.5, f"waveform unavailable: {exc}", transform=audio_axis.transAxes)
    audio_spans = np.asarray(raw_audio.get("spans", []), dtype=np.float64)
    audio_mask = np.asarray(raw_audio.get("mask", []), dtype=np.uint8)
    audio_step = max(1, len(audio_spans) // 300)
    for span, valid in zip(audio_spans[::audio_step], audio_mask[::audio_step]):
        if valid and span[1] > span[0]:
            audio_axis.axvspan(span[0], span[1], color="tab:orange", alpha=0.025)
    audio_axis.set_ylabel("audio")
    audio_axis.set_title("Decoded waveform and WavLM valid time spans")

    video_axis = axes[2]
    vision_spans = np.asarray(raw_vision.get("spans", []), dtype=np.float64)
    vision_mask = np.asarray(raw_vision.get("mask", []), dtype=np.uint8)
    frame_indices = np.asarray(raw_vision.get("frame_indices", []), dtype=np.int32)
    face_bbox = np.asarray(raw_vision.get("face_bbox", []), dtype=np.float32)
    if len(vision_spans):
        vision_centers = vision_spans.mean(axis=1)
        thumbnails = _load_keyframes(source_path, frame_indices, vision_centers, face_bbox, duration) if source_path else []
        thumb_width = max(duration / 35.0, 0.10)
        for center, image, frame_index in thumbnails:
            video_axis.imshow(
                image,
                extent=(max(0.0, center - thumb_width), min(duration, center + thumb_width), 0.06, 0.93),
                aspect="auto",
                interpolation="bilinear",
            )
            video_axis.text(center, 0.97, f"f{frame_index}", ha="center", va="top", fontsize=7)
        vision_step = max(1, len(vision_spans) // 300)
        for span, valid in zip(vision_spans[::vision_step], vision_mask[::vision_step]):
            if valid and span[1] > span[0]:
                video_axis.axvspan(span[0], span[1], color="tab:green", alpha=0.025)
    video_axis.set_ylim(0, 1)
    video_axis.set_yticks([0.5], ["video frames\nface box"])
    video_axis.set_title("Video keyframes on the same alignment timeline (green boxes: detected face)")

    mask_axis = axes[3]
    mask_axis.step(centers, masks["text"], where="mid", label="text")
    mask_axis.step(centers, masks["audio"] + 1.1, where="mid", label="audio")
    mask_axis.step(centers, masks["vision"] + 2.2, where="mid", label="vision")
    mask_axis.step(centers, text_low + 3.3, where="mid", label="text low-confidence", linestyle="--")
    mask_axis.set_yticks([0.5, 1.6, 2.7, 3.8], ["text", "audio", "vision", "text low"])
    mask_axis.set_ylabel("mask")
    mask_axis.set_title("50-bin masks and low-confidence text positions")
    mask_axis.legend(loc="upper right", ncol=4, fontsize=8)

    signal_axis = axes[4]
    signal_axis.plot(centers, text_confidence, label="text confidence")
    signal_axis.plot(centers, emotion.max(axis=1), label="max face emotion")
    text_norm = np.linalg.norm(text, axis=1)
    audio_norm = np.linalg.norm(audio, axis=1)
    vision_norm = np.linalg.norm(vision, axis=1)
    signal_axis.plot(centers, text_norm / max(float(text_norm.max()), 1e-6), label="text norm (scaled)")
    signal_axis.plot(centers, audio_norm / max(float(audio_norm.max()), 1e-6), label="audio norm (scaled)")
    signal_axis.plot(centers, vision_norm / max(float(vision_norm.max()), 1e-6), label="vision norm (scaled)")
    signal_axis.set_ylim(-0.02, 1.05)
    signal_axis.set_xlabel("clip time (s)")
    signal_axis.set_ylabel("value")
    signal_axis.set_title("Confidence, auxiliary emotion and aligned feature signals")
    signal_axis.legend(loc="upper right", ncol=3, fontsize=8)

    for axis in axes:
        for edge in grid[:, 0]:
            axis.axvline(float(edge), color="gray", linewidth=0.25, alpha=0.25)
        axis.grid(True, alpha=0.20)
        axis.set_xlim(0.0, max(duration, 1e-3))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
