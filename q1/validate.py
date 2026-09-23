from __future__ import annotations

import argparse
import csv
import pickle
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PACKAGE_REQUIRED_PATHS = [
    "README.md",
    "package_manifest.json",
    "q1_submission/q1_aligned_50.pkl",
    "q1_submission/manifest.csv",
    "q1_submission/alignment_log.csv",
    "q1_submission/duration_audit.csv",
    "q1_submission/text_review.csv",
    "q1_submission/text_audit_summary.csv",
    "q1_submission/validation_report.json",
    "code/q1/pipeline.py",
    "code/q1/validate.py",
    "code/q1/utils.py",
    "config/q1.yaml",
    "environment/environment.json",
    "environment/pip_freeze.txt",
    "tools/run_q1.sh",
    "tools/validate_q1.sh",
    "audit/typical_samples.json",
]


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_cell(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _cell_equal(left: Any, right: Any, tolerance: float = 1e-5) -> bool:
    left_number = _float_cell(left)
    right_number = _float_cell(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= tolerance
    return str(left) == str(right)


def _validate_cross_artifacts(
    artifact: dict[str, Any],
    ids: list[str],
    manifest_path: Path | None,
    alignment_log_path: Path | None,
    text_review_path: Path | None,
    duration_audit_path: Path | None,
) -> dict[str, Any]:
    """Check that the PKL, CSV audits, and per-sample metadata describe the same rows."""
    errors: list[str] = []
    warnings: list[str] = []
    n = len(ids)
    tables: dict[str, list[dict[str, str]]] = {}
    paths = {
        "manifest": manifest_path,
        "alignment_log": alignment_log_path,
        "text_review": text_review_path,
        "duration_audit": duration_audit_path,
    }
    for name, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            errors.append(f"Missing {name} file: {path}")
            continue
        try:
            tables[name] = _read_csv_rows(path)
        except Exception as exc:
            errors.append(f"Could not read {name} file: {exc}")

    expected_ids = [str(value) for value in ids]
    order_report: dict[str, int] = {}
    for name, rows in tables.items():
        row_ids = [str(row.get("id", "")) for row in rows]
        mismatch_count = sum(index >= len(row_ids) or row_ids[index] != sample_id for index, sample_id in enumerate(expected_ids))
        mismatch_count += max(0, len(row_ids) - n)
        order_report[name] = int(mismatch_count)
        if len(rows) != n:
            errors.append(f"{name} row count {len(rows)} != {n}")
        if mismatch_count:
            errors.append(f"{name} IDs/order differ from artifact: {mismatch_count} mismatches")

    manifest = tables.get("manifest", [])
    alignment = tables.get("alignment_log", [])
    text_review = tables.get("text_review", [])
    duration_audit = tables.get("duration_audit", [])
    metadata = artifact.get("metadata", [])
    metadata_mismatches: Counter[str] = Counter()
    metadata_specs = {
        "video_id": ("video_id", "video_id"),
        "clip_id": ("clip_id", "clip_id"),
        "video_path": ("video_path", "video_path_relative"),
        "sha256": ("sha256", "sha256"),
        "duration_container": ("duration_container", "duration_container"),
        "duration_alignment": ("duration_alignment", "duration_alignment"),
        "duration_source": ("duration_source", "duration_source"),
        "text_fallback_ratio": ("text_fallback_ratio", "text_fallback_ratio"),
        "text_review_status": ("text_review_status", "text_review_status"),
        "vision_valid_ratio": ("vision_valid_ratio", "vision_valid_ratio"),
        "vision_status": ("vision_status", "vision_status"),
        "quality_status": ("quality_status", "quality_status"),
        "processing_status": ("processing_status", "processing_status"),
    }
    if len(metadata) != n:
        errors.append(f"artifact metadata length {len(metadata)} != {n}")
    for index in range(min(n, len(metadata), len(manifest))):
        item = metadata[index] if isinstance(metadata[index], dict) else {}
        row = manifest[index]
        for field, (metadata_key, row_key) in metadata_specs.items():
            if not _cell_equal(item.get(metadata_key), row.get(row_key)):
                metadata_mismatches[field] += 1
    if metadata_mismatches:
        errors.append("artifact metadata does not match manifest.csv: " + ", ".join(
            f"{key}={value}" for key, value in sorted(metadata_mismatches.items())
        ))

    artifact_field_mismatches: Counter[str] = Counter()
    artifact_lists = {
        "processing_status": artifact.get("processing_status", []),
        "quality_status": artifact.get("quality_status", []),
        "vision_status": artifact.get("vision_status", []),
    }
    for field, values in artifact_lists.items():
        values = list(values)
        if len(values) != n:
            errors.append(f"artifact {field} length {len(values)} != {n}")
            continue
        for index, row in enumerate(manifest[:n]):
            if str(values[index]) != str(row.get(field, "")):
                artifact_field_mismatches[field] += 1
    for field, array_key in (("duration_container", "duration_container"), ("duration_alignment", "duration_alignment")):
        values = np.asarray(artifact.get(array_key, []), dtype=np.float64)
        if values.shape != (n,):
            continue
        for index, row in enumerate(manifest[:n]):
            if not _cell_equal(values[index], row.get(field)):
                artifact_field_mismatches[field] += 1
    if artifact_field_mismatches:
        errors.append("artifact arrays/statuses do not match manifest.csv: " + ", ".join(
            f"{key}={value}" for key, value in sorted(artifact_field_mismatches.items())
        ))

    text_mismatches: Counter[str] = Counter()
    for index in range(min(n, len(manifest), len(alignment), len(text_review))):
        manifest_row = manifest[index]
        alignment_row = alignment[index]
        review_row = text_review[index]
        for field in ("text_fallback_count", "text_fallback_ratio", "text_review_status"):
            review_key = {
                "text_fallback_count": "fallback_count",
                "text_fallback_ratio": "fallback_ratio",
                "text_review_status": "review_status",
            }[field]
            if not _cell_equal(manifest_row.get(field), review_row.get(review_key)):
                text_mismatches[field] += 1
        if not _cell_equal(manifest_row.get("text_review_status"), alignment_row.get("text_review_status")):
            text_mismatches["alignment_log.text_review_status"] += 1
        if not _cell_equal(manifest_row.get("text_invalid_timestamp_count"), alignment_row.get("text_invalid_timestamp_count")):
            text_mismatches["text_invalid_timestamp_count"] += 1
        try:
            fallback_ratio = float(review_row.get("fallback_ratio", 0.0))
            lexical_count = int(float(review_row.get("lexical_asr_word_count", 0.0)))
            invalid_count = int(float(review_row.get("invalid_timestamp_count", 0.0)))
            if fallback_ratio >= 0.999999:
                expected_status = "asr_empty_or_nonlexical_manual_review" if lexical_count == 0 else "asr_text_mismatch_manual_review"
            elif fallback_ratio >= 0.50:
                expected_status = "high_fallback_manual_review"
            elif invalid_count:
                expected_status = "partial_invalid_timestamp_manual_review"
            else:
                expected_status = "normal"
            if str(review_row.get("review_status", "")) != expected_status:
                text_mismatches["review_status_rule"] += 1
        except (TypeError, ValueError):
            text_mismatches["review_status_rule"] += 1
    if text_mismatches:
        errors.append("text audit fields/statuses are inconsistent: " + ", ".join(
            f"{key}={value}" for key, value in sorted(text_mismatches.items())
        ))

    duration_report = {
        "sample_count": len(duration_audit),
        "endpoint_status_counts": dict(Counter(row.get("duration_endpoint_status", "missing") for row in duration_audit)),
        "max_alignment_minus_decoded_video": 0.0,
        "max_alignment_minus_decoded_audio": 0.0,
        "max_alignment_minus_decoded_max": 0.0,
        "endpoint_rule_errors": [],
    }
    for row in duration_audit:
        try:
            video_gap = float(row.get("duration_alignment_minus_decoded_video", 0.0))
            audio_gap = float(row.get("duration_alignment_minus_decoded_audio", 0.0))
            max_gap = float(row.get("duration_alignment_minus_decoded_max", 0.0))
            video_tolerance = float(row.get("duration_endpoint_video_tolerance", 0.10))
            audio_tolerance = float(row.get("duration_endpoint_audio_tolerance", 0.25))
            negative_tolerance = float(row.get("duration_endpoint_negative_tolerance", 0.02))
        except (TypeError, ValueError) as exc:
            duration_report["endpoint_rule_errors"].append(f"{row.get('id', 'unknown')}: invalid endpoint value ({exc})")
            continue
        duration_report["max_alignment_minus_decoded_video"] = max(
            duration_report["max_alignment_minus_decoded_video"], video_gap
        )
        duration_report["max_alignment_minus_decoded_audio"] = max(
            duration_report["max_alignment_minus_decoded_audio"], audio_gap
        )
        duration_report["max_alignment_minus_decoded_max"] = max(
            duration_report["max_alignment_minus_decoded_max"], max_gap
        )
        sample_id = row.get("id", "unknown")
        if video_gap < -negative_tolerance:
            duration_report["endpoint_rule_errors"].append(f"{sample_id}: decoded video end exceeds alignment")
        if audio_gap < -negative_tolerance:
            duration_report["endpoint_rule_errors"].append(f"{sample_id}: decoded audio end exceeds alignment")
        if video_gap > video_tolerance:
            duration_report["endpoint_rule_errors"].append(f"{sample_id}: video endpoint gap {video_gap:.6f} > {video_tolerance:.6f}")
        if audio_gap > audio_tolerance:
            duration_report["endpoint_rule_errors"].append(f"{sample_id}: audio endpoint gap {audio_gap:.6f} > {audio_tolerance:.6f}")
        if row.get("duration_endpoint_status") == "manual_review" or row.get("duration_endpoint_issues"):
            duration_report["endpoint_rule_errors"].append(
                f"{sample_id}: recorded endpoint status/issues={row.get('duration_endpoint_status')}/{row.get('duration_endpoint_issues')}"
            )
    if duration_audit_path is not None and len(duration_audit) != n:
        errors.append(f"duration_audit row count {len(duration_audit)} != {n}")
    if duration_audit_path is not None and duration_report["endpoint_rule_errors"]:
        errors.extend(duration_report["endpoint_rule_errors"])

    quality_counts = {
        "quality_status": dict(Counter(str(value) for value in artifact.get("quality_status", []))),
        "vision_status": dict(Counter(str(value) for value in artifact.get("vision_status", []))),
    }
    quality_counts["text_review_status"] = dict(Counter(row.get("text_review_status", "") for row in manifest))
    if quality_counts["quality_status"].get("missing", 0) or quality_counts["vision_status"].get("missing", 0):
        warnings.append("quality status contains missing modalities; retain masks/statuses in downstream interpretation")
    if any(row.get("text_review_status") != "normal" for row in manifest):
        warnings.append("text audit contains manual-review samples; do not treat all word timestamps as equally reliable")

    structure_conclusion = "passed" if not errors else "failed"
    quality_conclusion = "conditional_pass" if not errors else "blocked_by_structural_errors"
    return {
        "errors": errors,
        "warnings": warnings,
        "table_order_mismatch_counts": order_report,
        "metadata_mismatch_counts": dict(metadata_mismatches),
        "artifact_field_mismatch_counts": dict(artifact_field_mismatches),
        "text_audit_mismatch_counts": dict(text_mismatches),
        "duration_endpoint_audit": duration_report,
        "quality_counts": quality_counts,
        "conclusions": {
            "structure": structure_conclusion,
            "quality": quality_conclusion,
            "reproducibility_package": "pending",
            "overall": "pending_reproducibility_package" if structure_conclusion == "passed" else "blocked",
        },
    }


def validate_submission_package(zip_path: Path, required_paths: list[str]) -> dict[str, Any]:
    """Validate the compact reproducibility archive without unpacking model weights."""
    errors: list[str] = []
    warnings: list[str] = []
    if not zip_path.exists():
        return {"zip": str(zip_path), "errors": [f"Package not found: {zip_path}"], "warnings": []}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = sorted(name for name in archive.namelist() if not name.endswith("/"))
            bad_member = archive.testzip()
            if bad_member:
                errors.append(f"Package CRC check failed for {bad_member}")
    except (OSError, zipfile.BadZipFile) as exc:
        return {"zip": str(zip_path), "errors": [f"Could not inspect package: {exc}"], "warnings": []}
    name_set = set(names)
    missing = [path for path in required_paths if path not in name_set]
    if missing:
        errors.append("Package is missing required files: " + ", ".join(missing))
    forbidden = [name for name in names if name.endswith("q1_unaligned.pkl") or "/models/" in f"/{name}" or name.startswith("models/")]
    if forbidden:
        errors.append("Package contains local audit/model payloads that should remain outside the compact archive: " + ", ".join(forbidden))
    if not any(name.lower().endswith(".png") for name in names):
        errors.append("Package contains no typical-sample figure")
    return {
        "zip": str(zip_path),
        "file_count": len(names),
        "files": names,
        "required_paths": required_paths,
        "missing_required_paths": missing,
        "errors": errors,
        "warnings": warnings,
    }


def validate_artifact(
    artifact_path: Path,
    expected_count: int = 100,
    expected_dims: tuple[int, int, int] = (768, 768, 768),
    require_complete: bool = True,
    manifest_path: Path | None = None,
    raw_audit_path: Path | None = None,
    alignment_log_path: Path | None = None,
    text_review_path: Path | None = None,
    duration_audit_path: Path | None = None,
    package_zip_path: Path | None = None,
    package_required_paths: list[str] | None = None,
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
    cross_report = _validate_cross_artifacts(
        artifact,
        [str(value) for value in ids],
        manifest_path,
        alignment_log_path,
        text_review_path,
        duration_audit_path,
    )
    errors.extend(cross_report["errors"])
    warnings.extend(cross_report["warnings"])
    package_report = None
    conclusions = dict(cross_report["conclusions"])
    if package_zip_path is not None:
        package_report = validate_submission_package(package_zip_path, package_required_paths or [])
        errors.extend(package_report["errors"])
        warnings.extend(package_report["warnings"])
        conclusions["reproducibility_package"] = "passed" if not package_report["errors"] else "failed"
        conclusions["overall"] = (
            "passed_with_quality_conditions"
            if conclusions["structure"] == "passed" and conclusions["reproducibility_package"] == "passed"
            else "blocked"
        )
    return {
        "artifact": str(artifact_path),
        "schema_version": artifact.get("schema_version", ""),
        "sample_count": n,
        "errors": errors,
        "warnings": warnings,
        "raw_audit": raw_report,
        "cross_artifact": cross_report,
        "package": package_report,
        "conclusions": conclusions,
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
    parser.add_argument("--alignment-log")
    parser.add_argument("--text-review")
    parser.add_argument("--duration-audit")
    parser.add_argument("--package-zip")
    args = parser.parse_args()
    result = validate_artifact(
        Path(args.artifact),
        expected_count=args.expected_count,
        require_complete=not args.allow_failures,
        manifest_path=Path(args.manifest) if args.manifest else None,
        raw_audit_path=Path(args.raw_audit) if args.raw_audit else None,
        alignment_log_path=Path(args.alignment_log) if args.alignment_log else None,
        text_review_path=Path(args.text_review) if args.text_review else None,
        duration_audit_path=Path(args.duration_audit) if args.duration_audit else None,
        package_zip_path=Path(args.package_zip) if args.package_zip else None,
        package_required_paths=DEFAULT_PACKAGE_REQUIRED_PATHS,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
