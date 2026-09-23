from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import align_text_words, make_time_grid, overlap_pool, standardize_aligned
from .config import Q1Config, dump_yaml, load_config
from .extractors import ASRAligner, AudioExtractor, TextExtractor, VisionExtractor
from .utils import (
    atomic_pickle_dump,
    command_version,
    configure_logging,
    environment_snapshot,
    extract_audio,
    probe_media,
    set_deterministic,
    sha256_file,
    verify_model_manifest,
    write_csv,
    write_json,
)


def _normal_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _string_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _column_index(headers: list[Any], names: list[str]) -> int:
    normalized = {_normal_header(value): index for index, value in enumerate(headers)}
    for name in names:
        key = _normal_header(name)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not find any of {names} in columns {headers}")


def read_label_table(path: Path) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active
    rows = list(worksheet.iter_rows(values_only=True))
    header_row = next((index for index, row in enumerate(rows) if any(value is not None for value in row)), None)
    if header_row is None:
        raise ValueError(f"Empty label table: {path}")
    headers = list(rows[header_row])
    indices = {
        "video_id": _column_index(headers, ["video_id", "video id"]),
        "clip_id": _column_index(headers, ["clip_id", "clip id"]),
        "text": _column_index(headers, ["text", "raw_text", "raw text"]),
        "label": _column_index(headers, ["label", "regression_label", "regression label"]),
        "annotation": _column_index(headers, ["annotation", "classification", "classification_label"]),
    }
    records = []
    for row in rows[header_row + 1 :]:
        if not any(value is not None for value in row):
            continue
        video_id = _string_id(row[indices["video_id"]])
        clip_id = _string_id(row[indices["clip_id"]])
        if not video_id or not clip_id:
            continue
        label_value = row[indices["label"]]
        try:
            label = float(label_value) if label_value is not None else None
        except (TypeError, ValueError):
            label = None
        records.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "id": f"{video_id}$_${clip_id}",
                "raw_text": str(row[indices["text"]] or ""),
                "label": label,
                "annotation": str(row[indices["annotation"]] or ""),
            }
        )
    if len(records) != 100:
        raise ValueError(f"Expected 100 label rows, found {len(records)} in {path}")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs in label table")
    return records


def build_manifest(config: Q1Config, logger: logging.Logger) -> list[dict[str, Any]]:
    label_file = config.path_for("label_file")
    video_root = config.path_for("video_root")
    labels = read_label_table(label_file)
    video_map = {
        (path.parent.name, path.stem): path
        for path in video_root.rglob("*.mp4")
    }
    if len(video_map) != 100:
        raise ValueError(f"Expected exactly 100 MP4 files under {video_root}, found {len(video_map)}")
    records = []
    for label in labels:
        key = (label["video_id"], label["clip_id"])
        video_path = video_map.get(key)
        if video_path is None:
            raise FileNotFoundError(f"No video for {label['id']}: {key}")
        media = probe_media(video_path)
        if media["duration"] <= 0:
            raise ValueError(f"Invalid video duration for {video_path}")
        record = {
            **label,
            "video_path": str(video_path),
            "video_path_relative": str(video_path.relative_to(config.project_root)),
            "duration": float(media["duration"]),
            "video_fps": float(media["video_fps"]),
            "video_width": int(media["video_width"]),
            "video_height": int(media["video_height"]),
            "video_codec": media["video_codec"],
            "audio_codec": media["audio_codec"],
            "audio_sample_rate_original": media["audio_sample_rate_original"],
            "has_audio": int(media["has_audio"]),
            "sha256": sha256_file(video_path),
            "feature_status": "indexed",
        }
        records.append(record)
    logger.info("Manifest validated: %d labels, %d videos, %d unique IDs", len(labels), len(video_map), len({x['id'] for x in records}))
    return records


def _safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)


def _empty_result(config: Q1Config, entry: dict[str, Any], reason: str) -> dict[str, Any]:
    length = config.aligned_length
    text_dim, audio_dim, vision_dim = config.main_feature_dims
    return {
        "text": np.zeros((length, text_dim), dtype=np.float32),
        "audio": np.zeros((length, audio_dim), dtype=np.float32),
        "vision": np.zeros((length, vision_dim), dtype=np.float32),
        "text_mask": np.zeros(length, dtype=np.uint8),
        "audio_mask": np.zeros(length, dtype=np.uint8),
        "vision_mask": np.zeros(length, dtype=np.uint8),
        "vision_emotion_probs": np.zeros((length, 7), dtype=np.float32),
        "audio_prosody": np.zeros((length, 5), dtype=np.float32),
        "audio_prosody_mask": np.zeros(length, dtype=np.uint8),
        "duration": float(entry["duration"]),
        "time_grid": make_time_grid(float(entry["duration"]), length),
        "raw_lengths": {"text": 0, "audio": 0, "vision": 0},
        "valid_lengths": {"text": 0, "audio": 0, "vision": 0},
        "stats": {
            "text_valid_ratio": 0.0,
            "audio_valid_ratio": 0.0,
            "vision_valid_ratio": 0.0,
            "failure_reason": reason,
        },
        "raw_audit": {"id": entry["id"], "error": reason},
        "status": "failed",
        "failure_reason": reason,
    }


def process_sample(
    config: Q1Config,
    entry: dict[str, Any],
    text_extractor: TextExtractor,
    asr_aligner: ASRAligner,
    audio_extractor: AudioExtractor,
    vision_extractor: VisionExtractor,
) -> dict[str, Any]:
    duration = float(entry["duration"])
    grid = make_time_grid(duration, config.aligned_length)
    audio = extract_audio(Path(entry["video_path"]), int(config.get("audio_sample_rate", 16000)))
    asr_words = asr_aligner.extract(audio, duration)
    text_raw = text_extractor.extract(entry["raw_text"])
    aligned_words = align_text_words(
        entry["raw_text"],
        asr_words,
        duration,
        max_edit_ratio=float(config.get("text_match_max_edit_ratio", 0.34)),
        fallback_confidence=float(config.get("text_fallback_confidence", 0.20)),
    )
    text_spans = np.asarray(
        [[aligned_words[index]["start"], aligned_words[index]["end"]] for index in text_raw.word_indices],
        dtype=np.float64,
    ).reshape((-1, 2))
    text_quality = np.asarray([aligned_words[index]["confidence"] for index in text_raw.word_indices], dtype=np.float64)
    text_aligned, text_mask, text_weights = overlap_pool(
        text_raw.features, text_spans, grid, quality=text_quality
    )

    audio_raw = audio_extractor.extract(audio, duration)
    audio_aligned, audio_mask, audio_weights = overlap_pool(
        audio_raw.features,
        audio_raw.spans,
        grid,
        quality=audio_raw.mask,
        valid_mask=audio_raw.mask.astype(bool),
    )
    prosody_aligned, prosody_mask, _ = overlap_pool(
        audio_raw.prosody,
        audio_raw.spans,
        grid,
        quality=audio_raw.prosody_mask,
        valid_mask=audio_raw.prosody_mask.astype(bool),
    )

    vision_raw = vision_extractor.extract(Path(entry["video_path"]), duration)
    vision_aligned, vision_mask, vision_weights = overlap_pool(
        vision_raw.features,
        vision_raw.spans,
        grid,
        quality=vision_raw.quality,
        valid_mask=vision_raw.mask.astype(bool),
    )
    emotion_aligned, emotion_mask, _ = overlap_pool(
        vision_raw.emotion_probs,
        vision_raw.spans,
        grid,
        quality=vision_raw.quality,
        valid_mask=vision_raw.mask.astype(bool),
    )
    vision_mask = np.minimum(vision_mask, emotion_mask).astype(np.uint8)

    valid_lengths = {
        "text": int(text_mask.sum()),
        "audio": int(audio_mask.sum()),
        "vision": int(vision_mask.sum()),
    }
    stats = {
        "text_valid_ratio": float(text_mask.mean()),
        "audio_valid_ratio": float(audio_mask.mean()),
        "vision_valid_ratio": float(vision_mask.mean()),
        "text_word_count": len(aligned_words),
        "text_asr_word_count": len(asr_words),
        "text_exact_match_count": sum(item["match_type"] == "exact" for item in aligned_words),
        "text_edit_match_count": sum(item["match_type"] == "edit_distance" for item in aligned_words),
        "text_fallback_count": sum(item["fallback"] for item in aligned_words),
        "audio_raw_frame_count": int(len(audio_raw.features)),
        "vision_raw_frame_count": int(vision_raw.frame_count),
        "vision_face_frame_count": int(vision_raw.detector_stats["face_frames"]),
        "vision_face_detection_ratio": float(vision_raw.detector_stats["face_detection_ratio"]),
        "vision_track_switches": int(vision_raw.detector_stats["track_switches"]),
        "text_weight_sum": float(text_weights.sum()),
        "audio_weight_sum": float(audio_weights.sum()),
        "vision_weight_sum": float(vision_weights.sum()),
        "wavlm_effective_stride_samples": int(audio_raw.timing.get("effective_stride_samples", 0)),
        "wavlm_receptive_field_samples": int(audio_raw.timing.get("receptive_field_samples", 0)),
    }
    raw_audit = {
        "id": entry["id"],
        "text": {
            "features": text_raw.features.astype(np.float16),
            "spans": text_spans.astype(np.float32),
            "words": aligned_words,
            "asr_words": asr_words,
            "overflow_chunks": text_raw.overflow_chunks,
        },
        "audio": {
            "features": audio_raw.features.astype(np.float16),
            "spans": audio_raw.spans.astype(np.float32),
            "mask": audio_raw.mask,
            "prosody": audio_raw.prosody.astype(np.float32),
            "prosody_mask": audio_raw.prosody_mask,
            "timing": audio_raw.timing,
        },
        "vision": {
            "features": vision_raw.features.astype(np.float16),
            "emotion_probs": vision_raw.emotion_probs.astype(np.float32),
            "spans": vision_raw.spans.astype(np.float32),
            "mask": vision_raw.mask,
            "quality": vision_raw.quality,
            "face_bbox": vision_raw.face_bbox,
            "face_confidence": vision_raw.face_confidence,
            "track_ids": vision_raw.track_ids,
            "frame_indices": vision_raw.frame_indices,
        },
    }
    return {
        "text": text_aligned,
        "audio": audio_aligned,
        "vision": vision_aligned,
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "vision_mask": vision_mask,
        "vision_emotion_probs": emotion_aligned,
        "audio_prosody": prosody_aligned,
        "audio_prosody_mask": prosody_mask,
        "duration": duration,
        "time_grid": grid,
        "raw_lengths": {"text": int(len(text_raw.features)), "audio": int(len(audio_raw.features)), "vision": int(vision_raw.frame_count)},
        "valid_lengths": valid_lengths,
        "stats": stats,
        "raw_audit": raw_audit,
        "status": "complete",
        "failure_reason": "",
    }


def _write_audit_files(audit_dir: Path, result: dict[str, Any]) -> None:
    sample_id = result["raw_audit"]["id"]
    safe = _safe_name(sample_id)
    timestamp_dir = audit_dir / "raw_timestamps"
    face_dir = audit_dir / "face_tracks"
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    raw = result["raw_audit"]
    write_json(
        {
            "id": sample_id,
            "text_words": raw.get("text", {}).get("words", []),
            "asr_words": raw.get("text", {}).get("asr_words", []),
            "audio_spans": raw.get("audio", {}).get("spans", []),
            "vision_spans": raw.get("vision", {}).get("spans", []),
        },
        timestamp_dir / f"{safe}.json",
    )
    write_json(
        {
            "id": sample_id,
            "face_bbox": raw.get("vision", {}).get("face_bbox", []),
            "face_confidence": raw.get("vision", {}).get("face_confidence", []),
            "mask": raw.get("vision", {}).get("mask", []),
            "track_ids": raw.get("vision", {}).get("track_ids", []),
        },
        face_dir / f"{safe}.json",
    )


def _submission_readme(config: Q1Config, runtime: dict[str, Any], run_id: str) -> str:
    return f"""# Question 1 feature artifact

This directory is generated from the 100 raw videos in attachment 1. It is an independent Question 1 interface with three 768-dimensional main features:

```text
text   [N, 50, 768]
audio  [N, 50, 768]
vision [N, 50, 768]
```

The Question 2 and Question 3 models must continue to use attachment 2's `768/74/35` interface. This artifact is not a replacement for attachment 2.

The provided transcript is retained as `raw_text`. Whisper word timestamps are used only to locate the provided words in time. The 50 bins are proportional half-open intervals over each clip duration, and each feature is pooled by temporal overlap and quality weight. Invalid bins are zero after standardization and have mask value 0.

Run ID: `{run_id}`  
Config hash: `{config.config_hash}`  
Model profile: `{config.get('model_profile', 'uniform_768_reproducible')}`

Files:

- `q1_aligned_50.pkl`: submission candidate with aligned features, masks, lengths, timestamps and metadata.
- `manifest.csv`: one row per source sample.
- `alignment_log.csv`: extraction, alignment and quality statistics.
- `normalization.json`: per-modality z-score parameters computed from valid Question 1 bins.
- `config.yaml`: resolved configuration and runtime versions.

The local audit directory contains unaligned sequences, raw timestamps, face tracks and detailed logs. It is intentionally excluded from the submission archive.
"""


def _zip_submission(submission_dir: Path, zip_path: Path) -> int:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(submission_dir.parent))
    return zip_path.stat().st_size


def run_pipeline(
    config: Q1Config,
    run_id: str,
    limit: int | None = None,
    sample_id: str | None = None,
    allow_failures: bool = False,
    skip_model_hash: bool = False,
) -> Path:
    run_dir = config.path_for("run_root") / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    submission_dir = run_dir / "outputs" / "q1_submission"
    audit_dir = run_dir / "outputs" / "q1_audit"
    log_path = run_dir / "logs" / f"q1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(log_path)
    (run_dir / "RUNNING").write_text("", encoding="utf-8")
    started = time.time()
    try:
        seed = int(config.get("random_seed", 2026))
        set_deterministic(seed)
        logger.info("Starting Question 1 run %s", run_id)
        environment = environment_snapshot(config.path_for("model_manifest"))
        write_json(environment, run_dir / "environment" / "environment.json")
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        (run_dir / "environment" / "pip_freeze.txt").write_text(
            pip_freeze.stdout + ("\n" + pip_freeze.stderr if pip_freeze.stderr else ""),
            encoding="utf-8",
        )
        model_check = verify_model_manifest(config, logger, verify_hashes=not skip_model_hash)
        write_json(model_check, run_dir / "environment" / "model_verification.json")
        manifest = build_manifest(config, logger)
        if sample_id:
            manifest = [entry for entry in manifest if entry["id"] == sample_id]
            if not manifest:
                raise KeyError(f"Unknown sample ID: {sample_id}")
        if limit is not None:
            manifest = manifest[: int(limit)]
        logger.info("Samples selected for run: %d", len(manifest))

        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("Inference device: %s", device)
        models_root = config.path_for("models_root")
        text_extractor = TextExtractor(config.path_for("text_model_path"), device, int(config.get("text_max_length", 8192)))
        asr_aligner = ASRAligner(
            config.path_for("asr_timestamp_model_path"),
            device,
            float(config.get("asr_window_s", 30.0)),
            float(config.get("asr_overlap_s", 5.0)),
        )
        audio_extractor = AudioExtractor(
            config.path_for("audio_model_path"), device, int(config.get("audio_sample_rate", 16000))
        )
        vision_extractor = VisionExtractor(
            config.path_for("vision_model_path"),
            device,
            float(config.get("video_output_fps", 25.0)),
            float(config.get("face_detector_min_confidence", 0.5)),
            float(config.get("face_crop_margin", 0.20)),
            float(config.get("face_track_iou_threshold", 0.20)),
            int(config.get("face_track_max_gap_frames", 2)),
        )
        del models_root

        results: list[dict[str, Any]] = []
        audit_results: list[dict[str, Any]] = []
        completed = 0
        failures = 0
        for index, entry in enumerate(manifest, start=1):
            logger.info("[%d/%d] Processing %s", index, len(manifest), entry["id"])
            try:
                result = process_sample(config, entry, text_extractor, asr_aligner, audio_extractor, vision_extractor)
                completed += 1
                logger.info(
                    "Completed %s: text=%.3f audio=%.3f vision=%.3f",
                    entry["id"],
                    result["stats"]["text_valid_ratio"],
                    result["stats"]["audio_valid_ratio"],
                    result["stats"]["vision_valid_ratio"],
                )
            except Exception as exc:  # preserve a traceable row for every source sample
                failures += 1
                reason = f"{type(exc).__name__}: {exc}"
                logger.error("Failed %s: %s\n%s", entry["id"], reason, traceback.format_exc())
                result = _empty_result(config, entry, reason)
            _write_audit_files(audit_dir, result)
            result["raw_audit"]["entry"] = {key: value for key, value in entry.items() if key != "video_path"}
            results.append(result)
            audit_results.append(result["raw_audit"])

        if not results:
            raise RuntimeError("No samples were processed")
        text_stack = np.stack([item["text"] for item in results]).astype(np.float32)
        audio_stack = np.stack([item["audio"] for item in results]).astype(np.float32)
        vision_stack = np.stack([item["vision"] for item in results]).astype(np.float32)
        text_masks = np.stack([item["text_mask"] for item in results])
        audio_masks = np.stack([item["audio_mask"] for item in results])
        vision_masks = np.stack([item["vision_mask"] for item in results])
        text_stack, text_mean, text_std, _ = standardize_aligned(text_stack, text_masks)
        audio_stack, audio_mean, audio_std, _ = standardize_aligned(audio_stack, audio_masks)
        vision_stack, vision_mean, vision_std, _ = standardize_aligned(vision_stack, vision_masks)
        submission_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": "q1.uniform_768.v1",
            "model_profile": config.get("model_profile", "uniform_768_reproducible"),
            "config_hash": config.config_hash,
            "id": [entry["id"] for entry in manifest],
            "text": text_stack.astype(np.float16),
            "audio": audio_stack.astype(np.float16),
            "vision": vision_stack.astype(np.float16),
            "auxiliary": {
                "vision_emotion_probs": np.stack([item["vision_emotion_probs"] for item in results]).astype(np.float32),
                "audio_prosody": np.stack([item["audio_prosody"] for item in results]).astype(np.float32),
                "audio_prosody_mask": np.stack([item["audio_prosody_mask"] for item in results]).astype(np.uint8),
            },
            "masks": {
                "text": text_masks.astype(np.uint8),
                "audio": audio_masks.astype(np.uint8),
                "vision": vision_masks.astype(np.uint8),
            },
            "durations": np.asarray([item["duration"] for item in results], dtype=np.float32),
            "time_grid": np.stack([item["time_grid"] for item in results]).astype(np.float32),
            "valid_lengths": {
                modality: np.asarray([item["valid_lengths"][modality] for item in results], dtype=np.int16)
                for modality in ("text", "audio", "vision")
            },
            "raw_lengths": {
                modality: np.asarray([item["raw_lengths"][modality] for item in results], dtype=np.int32)
                for modality in ("text", "audio", "vision")
            },
            "raw_text": [entry["raw_text"] for entry in manifest],
            "labels": np.asarray([entry["label"] if entry["label"] is not None else np.nan for entry in manifest], dtype=np.float32),
            "annotations": [entry["annotation"] for entry in manifest],
            "status": [item["status"] for item in results],
            "failure_reasons": [item["failure_reason"] for item in results],
            "metadata": [
                {
                    "video_id": entry["video_id"],
                    "clip_id": entry["clip_id"],
                    "video_path": entry["video_path_relative"],
                    "sha256": entry["sha256"],
                }
                for entry in manifest
            ],
        }
        artifact_path = submission_dir / "q1_aligned_50.pkl"
        atomic_pickle_dump(artifact, artifact_path)
        normalization = {
            "method": "zscore_on_valid_aligned_bins",
            "epsilon": 1e-6,
            "text": {"mean": text_mean, "std": text_std, "dim": 768},
            "audio": {"mean": audio_mean, "std": audio_std, "dim": 768},
            "vision": {"mean": vision_mean, "std": vision_std, "dim": 768},
        }
        write_json(normalization, submission_dir / "normalization.json")

        manifest_rows = []
        alignment_rows = []
        for entry, result in zip(manifest, results):
            entry = dict(entry)
            entry.update(
                {
                    "text_length": result["raw_lengths"]["text"],
                    "audio_length": result["raw_lengths"]["audio"],
                    "vision_length": result["raw_lengths"]["vision"],
                    "feature_dim_text": 768,
                    "feature_dim_audio": 768,
                    "feature_dim_vision": 768,
                    "aligned_length": config.aligned_length,
                    "alignment_granularity": "proportional_50_half_open_intervals",
                    "feature_status": result["status"],
                    "failure_reason": result["failure_reason"],
                }
            )
            manifest_rows.append(entry)
            alignment_row = {"id": entry["id"], "duration": entry["duration"], "status": result["status"], "failure_reason": result["failure_reason"]}
            alignment_row.update(result["stats"])
            alignment_rows.append(alignment_row)
        manifest_fields = [
            "id", "video_id", "clip_id", "video_path_relative", "duration", "raw_text", "label", "annotation",
            "audio_sample_rate_original", "video_fps", "video_width", "video_height", "text_length", "audio_length",
            "vision_length", "feature_dim_text", "feature_dim_audio", "feature_dim_vision", "aligned_length",
            "alignment_granularity", "feature_status", "failure_reason", "sha256",
        ]
        write_csv(manifest_rows, submission_dir / "manifest.csv", manifest_fields)
        alignment_fields = sorted({key for row in alignment_rows for key in row})
        write_csv(alignment_rows, submission_dir / "alignment_log.csv", alignment_fields)

        if bool(config.get("save_unaligned", True)):
            atomic_pickle_dump({"schema_version": "q1.unaligned.v1", "samples": audit_results}, audit_dir / "q1_unaligned.pkl")
        write_json(environment, audit_dir / "environment.json")
        resolved_config = dict(config.values)
        resolved_config["runtime"] = environment
        resolved_config["model_verification"] = model_check
        resolved_config["config_hash"] = config.config_hash
        dump_yaml(resolved_config, submission_dir / "config.yaml")
        (submission_dir / "README.md").write_text(_submission_readme(config, environment, run_id), encoding="utf-8")

        if bool(config.get("generate_typical_sample_figure", True)):
            try:
                from .visualize_sample import render_sample_figure

                quality = np.minimum.reduce(
                    [text_masks.mean(axis=1), audio_masks.mean(axis=1), vision_masks.mean(axis=1)]
                )
                typical_index = int(np.argmax(quality))
                render_sample_figure(
                    manifest[typical_index]["id"],
                    {
                        "time_grid": artifact["time_grid"][typical_index],
                        "text": artifact["text"][typical_index],
                        "audio": artifact["audio"][typical_index],
                        "vision": artifact["vision"][typical_index],
                        "masks": {key: value[typical_index] for key, value in artifact["masks"].items()},
                        "vision_emotion_probs": artifact["auxiliary"]["vision_emotion_probs"][typical_index],
                    },
                    audit_dir / "typical_sample_alignment.png",
                )
                write_json(
                    {
                        "id": manifest[typical_index]["id"],
                        "index": typical_index,
                        "combined_valid_ratio": float(quality[typical_index]),
                    },
                    audit_dir / "typical_sample.json",
                )
            except Exception as exc:
                logger.warning("Typical sample visualization was not generated: %s", exc)

        from .validate import validate_artifact

        validation = validate_artifact(
            artifact_path,
            expected_count=len(manifest),
            expected_dims=config.main_feature_dims,
            require_complete=bool(config.get("require_all_samples_complete", True)) and not allow_failures,
            manifest_path=submission_dir / "manifest.csv",
        )
        write_json(validation, submission_dir / "validation_report.json")
        if validation["errors"]:
            raise RuntimeError("Output validation failed: " + "; ".join(validation["errors"]))

        zip_path = run_dir / "outputs" / f"q1_submission_{run_id}.zip"
        zip_bytes = _zip_submission(submission_dir, zip_path)
        write_json(
            {
                "uncompressed_bytes": sum(path.stat().st_size for path in submission_dir.rglob("*") if path.is_file()),
                "compressed_bytes": zip_bytes,
                "limit_bytes": int(config.get("submission_max_bytes", 52_428_800)),
                "within_limit": zip_bytes <= int(config.get("submission_max_bytes", 52_428_800)),
            },
            submission_dir / "package_size.json",
        )
        if zip_bytes > int(config.get("submission_max_bytes", 52_428_800)):
            raise RuntimeError(f"Submission archive exceeds limit: {zip_bytes} bytes")

        elapsed = time.time() - started
        logger.info("Run complete: successful=%d failed=%d elapsed=%.1fs", completed, failures, elapsed)
        write_json(
            {
                "run_id": run_id,
                "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "successful_samples": completed,
                "failed_samples": failures,
                "elapsed_seconds": elapsed,
                "submission_zip": str(zip_path),
            },
            run_dir / "run_summary.json",
        )
        (run_dir / "RUNNING").unlink(missing_ok=True)
        (run_dir / "DONE").write_text("", encoding="utf-8")
        return run_dir
    except Exception:
        logger.error("Run failed:\n%s", traceback.format_exc())
        (run_dir / "RUNNING").unlink(missing_ok=True)
        (run_dir / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Question 1 raw-video feature extraction remotely.")
    parser.add_argument("--config", default="configs/q1.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--skip-model-hash", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(
        config,
        args.run_id,
        limit=args.limit,
        sample_id=args.sample_id,
        allow_failures=args.allow_failures,
        skip_model_hash=args.skip_model_hash,
    )


if __name__ == "__main__":
    main()
