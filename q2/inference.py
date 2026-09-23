from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import INPUT_DIMS, MAX_LENGTH, MODALITIES
from .data import NormalizationStats, SplitData, _extract_feature, sha256_file
from .evaluation import predict_split


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _unwrap_test_payload(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and "test" in payload and isinstance(payload["test"], Mapping):
        return payload["test"]
    if not isinstance(payload, Mapping):
        raise ValueError("Attachment 3 pickle must contain a mapping")
    return payload


def load_attachment3_samples(directory: str | Path, max_length: int = MAX_LENGTH) -> SplitData:
    directory = Path(directory).resolve()
    files = sorted(directory.glob("*.pkl"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No attachment 3 pickle files found in {directory}")
    feature_parts: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
    mask_parts: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
    ids: list[str] = []
    source_files: list[str] = []
    id_sources: list[str] = []
    text_bert_ignored = 0
    for file_path in files:
        import pickle

        with file_path.open("rb") as handle:
            raw = _unwrap_test_payload(pickle.load(handle))
        n = None
        for modality in MODALITIES:
            if modality in raw:
                candidate = np.asarray(raw[modality])
                if candidate.ndim >= 1:
                    n = int(candidate.shape[0])
                    break
        if n is None:
            # A text-only missing sample still has text_bert/raw_text metadata.
            n = int(np.asarray(raw.get("raw_text", [None])).reshape(-1).shape[0])
        if "text_bert" in raw and "text" not in raw:
            text_bert_ignored += n
        local_ids = raw.get("id")
        if local_ids is not None:
            values = list(np.asarray(local_ids).reshape(-1))
            if len(values) != n:
                raise ValueError(f"{file_path.name}: ID count {len(values)} != {n}")
            local_ids_text = [str(value) for value in values]
            local_id_source = "input_id_field"
        else:
            local_ids_text = [file_path.stem if n == 1 else f"{file_path.stem}__{index + 1:03d}" for index in range(n)]
            local_id_source = "source_file_name_fallback"
        for modality in MODALITIES:
            values, masks, _ = _extract_feature(raw, modality, n, max_length)
            feature_parts[modality].append(values)
            mask_parts[modality].append(masks)
        ids.extend(local_ids_text)
        source_files.extend([file_path.name] * n)
        id_sources.extend([local_id_source] * n)
    features = {modality: np.concatenate(feature_parts[modality], axis=0) for modality in MODALITIES}
    masks = {modality: np.concatenate(mask_parts[modality], axis=0) for modality in MODALITIES}
    return SplitData(
        name="attachment3",
        ids=ids,
        features=features,
        masks=masks,
        annotations=[None] * len(ids),
        mask_sources={modality: "zero_row_heuristic_or_explicit" for modality in MODALITIES},
        metadata={
            "source_directory": str(directory),
            "source_files": source_files,
            "id_sources": id_sources,
            "text_bert_ignored_count": text_bert_ignored,
            "input_file_count": len(files),
            "input_order": "natural filename order",
        },
    )


def _float_or_na(value: Any) -> str | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return number if np.isfinite(number) else "NA"


def write_attachment3_predictions(
    model: torch.nn.Module,
    samples: SplitData,
    stats: NormalizationStats,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    model_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = stats.transform(samples, name="attachment3_normalized")
    predictions = predict_split(model, normalized, device=device, batch_size=batch_size)
    prediction_path = output_dir / "q2_attachment3_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "predicted_annotation",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
                "predicted_intensity",
            ],
        )
        writer.writeheader()
        for index, sample_id in enumerate(samples.ids):
            predicted_class = int(predictions["predicted_class"][index])
            probabilities = predictions["probabilities"][index]
            writer.writerow(
                {
                    "id": sample_id,
                    "predicted_annotation": ("Negative", "Neutral", "Positive")[predicted_class],
                    "prob_negative": _float_or_na(probabilities[0]),
                    "prob_neutral": _float_or_na(probabilities[1]),
                    "prob_positive": _float_or_na(probabilities[2]),
                    "predicted_intensity": _float_or_na(predictions["intensity"][index]),
                }
            )

    audit_path = output_dir / "q2_attachment3_audit.csv"
    source_files = list(samples.metadata.get("source_files", []))
    id_sources = list(samples.metadata.get("id_sources", []))
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "id",
                "source_file",
                "id_source",
                "text_coverage",
                "audio_coverage",
                "vision_coverage",
                "all_unavailable",
                "max_probability",
                "nonfinite_output",
            ],
        )
        writer.writeheader()
        for index, sample_id in enumerate(samples.ids):
            probabilities = predictions["probabilities"][index]
            finite = bool(np.isfinite(probabilities).all() and np.isfinite(predictions["intensity"][index]))
            writer.writerow(
                {
                    "row_index": index,
                    "id": sample_id,
                    "source_file": source_files[index] if index < len(source_files) else "",
                    "id_source": id_sources[index] if index < len(id_sources) else "",
                    "text_coverage": float(predictions["coverage"][index, 0]),
                    "audio_coverage": float(predictions["coverage"][index, 1]),
                    "vision_coverage": float(predictions["coverage"][index, 2]),
                    "all_unavailable": bool(predictions["all_unavailable"][index]),
                    "max_probability": float(np.max(probabilities)) if finite else "NA",
                    "nonfinite_output": not finite,
                }
            )

    manifest = {
        "feature_version": "aligned_50",
        "prediction_file": str(prediction_path),
        "audit_file": str(audit_path),
        "sample_count": len(samples.ids),
        "id_coverage": len(samples.ids) == len(set(samples.ids)),
        "ids": samples.ids,
        "input_file_count": samples.metadata.get("input_file_count", 0),
        "input_order": samples.metadata.get("input_order"),
        "source_files": source_files,
        "id_sources": id_sources,
        "input_file_sha256": {
            name: sha256_file(Path(samples.metadata.get("source_directory", "")) / name)
            for name in sorted(set(source_files))
            if Path(samples.metadata.get("source_directory", ""), name).exists()
        },
        "text_bert_ignored_count": samples.metadata.get("text_bert_ignored_count", 0),
        "all_unavailable_count": int(predictions["all_unavailable"].sum()),
        "nonfinite_output_count": int(
            np.sum(~np.isfinite(predictions["probabilities"]).all(axis=1))
            + np.sum(~np.isfinite(predictions["intensity"]))
        ),
        "model_sha256": sha256_file(model_path) if model_path and Path(model_path).exists() else None,
        "normalization_sha256": sha256_file(normalization_path)
        if normalization_path and Path(normalization_path).exists()
        else None,
        "label_metrics": "not computed: attachment 3 has no labels",
    }
    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
