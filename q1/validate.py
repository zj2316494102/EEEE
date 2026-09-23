from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _raw_interval_check(
    spans: Any,
    duration: float,
    sample_id: str,
    modality: str,
    errors: list[str],
    counts: dict[str, int],
) -> None:
    array = np.asarray(spans, dtype=np.float32).astype(np.float64)
    if array.size == 0:
        return
    if array.ndim != 2 or array.shape[1] != 2:
        errors.append(f"{sample_id}/{modality} spans have invalid shape {array.shape}")
        return
    base_modality = "text" if modality.startswith("text") else modality
    counts[f"{base_modality}_interval_count"] += int(array.shape[0])
    if not np.isfinite(array).all():
        errors.append(f"{sample_id}/{modality} spans contain non-finite values")
    non_positive = int(np.sum(array[:, 1] <= array[:, 0]))
    out_of_range = int(np.sum((array[:, 0] < -1e-6) | (array[:, 1] > duration + 1e-4)))
    counts[f"{base_modality}_non_positive_count"] += non_positive
    counts[f"{base_modality}_out_of_range_count"] += out_of_range
    if non_positive:
        errors.append(f"{sample_id}/{modality} has {non_positive} non-positive intervals")
    if out_of_range:
        errors.append(f"{sample_id}/{modality} has {out_of_range} out-of-range intervals")


def validate_raw_audit(audit_path: Path, expected_count: int = 100) -> dict[str, Any]:
    """Validate persisted unaligned spans after float32 serialization."""
    errors: list[str] = []
    warnings: list[str] = []
    if not audit_path.exists():
        return {"audit": str(audit_path), "sample_count": 0, "errors": [f"Audit not found: {audit_path}"], "warnings": []}
    try:
        with audit_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        return {"audit": str(audit_path), "sample_count": 0, "errors": [f"Could not load audit: {exc}"], "warnings": []}
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    if len(samples) != expected_count:
        errors.append(f"Expected {expected_count} audit samples, found {len(samples)}")
    counts = {
        "text_interval_count": 0,
        "audio_interval_count": 0,
        "vision_interval_count": 0,
        "text_non_positive_count": 0,
        "audio_non_positive_count": 0,
        "vision_non_positive_count": 0,
        "text_out_of_range_count": 0,
        "audio_out_of_range_count": 0,
        "vision_out_of_range_count": 0,
        "invalid_text_timestamp_count": 0,
    }
    for sample in samples:
        sample_id = str(sample.get("id", "unknown"))
        entry = sample.get("entry", {})
        duration = float(entry.get("duration_alignment", entry.get("duration", 0.0)) or 0.0)
        text = sample.get("text", {}) or {}
        for index, word in enumerate(text.get("words", []) or []):
            start, end = word.get("start"), word.get("end")
            if start is None or end is None:
                counts["invalid_text_timestamp_count"] += 1
                if word.get("time_valid") or not word.get("invalid_reason"):
                    errors.append(f"{sample_id}/text word {index} has missing timestamp without invalid reason")
                continue
            _raw_interval_check([[start, end]], duration, sample_id, f"text_word_{index}", errors, counts)
        _raw_interval_check(text.get("spans", []), duration, sample_id, "text", errors, counts)
        _raw_interval_check((sample.get("audio", {}) or {}).get("spans", []), duration, sample_id, "audio", errors, counts)
        _raw_interval_check((sample.get("vision", {}) or {}).get("spans", []), duration, sample_id, "vision", errors, counts)
    return {
        "audit": str(audit_path),
        "schema_version": payload.get("schema_version", "") if isinstance(payload, dict) else "",
        "sample_count": len(samples),
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }


def validate_artifact(
    artifact_path: Path,
    expected_count: int = 100,
    expected_dims: tuple[int, int, int] = (768, 768, 768),
    require_complete: bool = True,
    manifest_path: Path | None = None,
    raw_audit_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not artifact_path.exists():
        return {"artifact": str(artifact_path), "errors": [f"Artifact not found: {artifact_path}"], "warnings": []}
    try:
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
    except Exception as exc:
        return {"artifact": str(artifact_path), "errors": [f"Could not load artifact: {exc}"], "warnings": []}

    ids = artifact.get("id")
    if not isinstance(ids, list):
        ids = list(ids) if ids is not None else []
    n = len(ids)
    if n != expected_count:
        errors.append(f"Expected {expected_count} samples, found {n}")
    if len(set(ids)) != n:
        errors.append("Sample IDs are not unique")
    for key in ("text", "audio", "vision"):
        if key not in artifact:
            errors.append(f"Missing main feature: {key}")
            continue
        array = np.asarray(artifact[key])
        expected_dim = expected_dims[("text", "audio", "vision").index(key)]
        if array.shape != (n, 50, expected_dim):
            errors.append(f"{key} shape {array.shape} != {(n, 50, expected_dim)}")
        if not np.isfinite(array).all():
            errors.append(f"{key} contains NaN or infinity")
    masks = artifact.get("masks", {})
    for key in ("text", "audio", "vision"):
        if key not in masks:
            errors.append(f"Missing mask: {key}")
            continue
        mask = np.asarray(masks[key])
        if mask.shape != (n, 50):
            errors.append(f"{key} mask shape {mask.shape} != {(n, 50)}")
        if not np.isin(mask, [0, 1]).all():
            errors.append(f"{key} mask contains values other than 0/1")
    durations = np.asarray(artifact.get("durations", []), dtype=np.float64)
    alignment_durations = np.asarray(artifact.get("duration_alignment", durations), dtype=np.float64)
    container_durations = np.asarray(artifact.get("duration_container", []), dtype=np.float64)
    time_grid = np.asarray(artifact.get("time_grid", []), dtype=np.float64)
    if durations.shape != (n,):
        errors.append(f"durations shape {durations.shape} != {(n,)}")
    if time_grid.shape != (n, 50, 2):
        errors.append(f"time_grid shape {time_grid.shape} != {(n, 50, 2)}")
    elif alignment_durations.shape == (n,):
        for index, (grid, duration) in enumerate(zip(time_grid, alignment_durations)):
            if not np.isfinite(grid).all():
                errors.append(f"time_grid contains non-finite values for sample {index}")
            if np.any(grid[:, 1] < grid[:, 0]):
                errors.append(f"time_grid interval reverses for sample {index}")
            if np.any(grid[1:, 0] < grid[:-1, 0]):
                errors.append(f"time_grid is not monotonic for sample {index}")
            if grid.size and (grid[0, 0] < -1e-6 or grid[-1, 1] > duration + 1e-4):
                errors.append(f"time_grid exceeds duration for sample {index}")
            if grid.size and abs(grid[0, 0]) > 1e-6:
                errors.append(f"time_grid does not start at zero for sample {index}")
    auxiliary = artifact.get("auxiliary", {})
    expected_aux = {
        "vision_emotion_probs": (n, 50, 7),
        "audio_prosody": (n, 50, 5),
        "audio_prosody_mask": (n, 50),
    }
    for key, shape in expected_aux.items():
        if key not in auxiliary:
            errors.append(f"Missing auxiliary field: {key}")
            continue
        array = np.asarray(auxiliary[key])
        if array.shape != shape:
            errors.append(f"Auxiliary {key} shape {array.shape} != {shape}")
        if not np.isfinite(array).all():
            errors.append(f"Auxiliary {key} contains NaN or infinity")
    text_confidence = np.asarray(artifact.get("text_confidence", []))
    text_low_confidence = np.asarray(artifact.get("text_low_confidence_mask", []))
    if text_confidence.shape != (n, 50):
        errors.append(f"text_confidence shape {text_confidence.shape} != {(n, 50)}")
    elif not np.isfinite(text_confidence).all():
        errors.append("text_confidence contains NaN or infinity")
    if text_low_confidence.shape != (n, 50):
        errors.append(f"text_low_confidence_mask shape {text_low_confidence.shape} != {(n, 50)}")
    elif not np.isin(text_low_confidence, [0, 1]).all():
        errors.append("text_low_confidence_mask contains values other than 0/1")
    if alignment_durations.shape != (n,):
        errors.append(f"duration_alignment shape {alignment_durations.shape} != {(n,)}")
    if container_durations.size and container_durations.shape != (n,):
        errors.append(f"duration_container shape {container_durations.shape} != {(n,)}")
    statuses = list(artifact.get("processing_status", artifact.get("status", [])))
    if len(statuses) != n:
        errors.append(f"status length {len(statuses)} != {n}")
    failed = [index for index, status in enumerate(statuses) if status != "complete"]
    if failed and require_complete:
        errors.append(f"{len(failed)} samples are not complete: {failed[:10]}")
    elif failed:
        warnings.append(f"{len(failed)} samples are not complete")
    reasons = list(artifact.get("failure_reasons", []))
    if len(reasons) != n:
        errors.append(f"failure_reasons length {len(reasons)} != {n}")
    for index in failed:
        if index >= len(reasons) or not reasons[index]:
            errors.append(f"Incomplete sample {index} has no failure reason")
    if manifest_path and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_ids = [row.get("id", "") for row in csv.DictReader(handle)]
        if manifest_ids != ids:
            errors.append("Artifact IDs do not match manifest.csv order")
    raw_report = None
    if raw_audit_path:
        raw_report = validate_raw_audit(raw_audit_path, expected_count=n)
        errors.extend(raw_report["errors"])
        warnings.extend(raw_report["warnings"])
    return {
        "artifact": str(artifact_path),
        "schema_version": artifact.get("schema_version", ""),
        "sample_count": n,
        "errors": errors,
        "warnings": warnings,
        "raw_audit": raw_report,
        "feature_shapes": {
            key: list(np.asarray(artifact[key]).shape) for key in ("text", "audio", "vision") if key in artifact
        },
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Question 1 aligned feature artifact.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--raw-audit")
    args = parser.parse_args()
    result = validate_artifact(
        Path(args.artifact),
        expected_count=args.expected_count,
        require_complete=not args.allow_failures,
        manifest_path=Path(args.manifest) if args.manifest else None,
        raw_audit_path=Path(args.raw_audit) if args.raw_audit else None,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
