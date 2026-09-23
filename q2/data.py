from __future__ import annotations

import csv
import ast
import hashlib
import pickle
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from . import INPUT_DIMS, MAX_LENGTH, MODALITIES


CLASS_NAMES = ("Negative", "Neutral", "Positive")
_XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class SplitData:
    """One aligned split with continuous features and explicit availability masks."""

    name: str
    ids: list[str]
    features: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    classification: np.ndarray | None = None
    regression: np.ndarray | None = None
    annotations: list[str | None] | None = None
    mask_sources: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.mask_sources = dict(self.mask_sources or {})
        self.metadata = dict(self.metadata or {})
        if self.annotations is None:
            self.annotations = [None] * len(self.ids)
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"{self.name} contains duplicate sample IDs")
        for modality in MODALITIES:
            if modality not in self.features or modality not in self.masks:
                raise ValueError(f"{self.name} is missing modality {modality}")
            values = np.asarray(self.features[modality])
            mask = np.asarray(self.masks[modality])
            expected = (len(self.ids), MAX_LENGTH, INPUT_DIMS[modality])
            if values.shape != expected:
                raise ValueError(f"{self.name}/{modality} shape {values.shape} != {expected}")
            if mask.shape != expected[:2]:
                raise ValueError(f"{self.name}/{modality} mask shape {mask.shape} != {expected[:2]}")
            if not np.isfinite(values).all():
                raise ValueError(f"{self.name}/{modality} contains non-finite values")
        if len(self.annotations or []) != len(self.ids):
            raise ValueError(f"{self.name} annotation length does not match IDs")
        if self.classification is not None:
            self.classification = np.asarray(self.classification, dtype=np.int64)
            if self.classification.shape != (len(self.ids),):
                raise ValueError(f"{self.name} classification label shape is invalid")
            if np.any((self.classification < 0) | (self.classification >= len(CLASS_NAMES))):
                raise ValueError(f"{self.name} contains class IDs outside 0..2")
        if self.regression is not None:
            self.regression = np.asarray(self.regression, dtype=np.float32)
            if self.regression.shape != (len(self.ids),):
                raise ValueError(f"{self.name} regression label shape is invalid")
            if not np.isfinite(self.regression).all():
                raise ValueError(f"{self.name} contains non-finite regression labels")

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def has_labels(self) -> bool:
        return self.classification is not None and self.regression is not None

    def mask_array(self) -> np.ndarray:
        return np.stack([np.asarray(self.masks[m], dtype=bool) for m in MODALITIES], axis=1)

    def feature_array(self) -> dict[str, np.ndarray]:
        return {m: np.asarray(self.features[m], dtype=np.float32) for m in MODALITIES}

    def subset(self, indices: Iterable[int] | slice, name: str | None = None) -> "SplitData":
        if isinstance(indices, slice):
            selected = np.arange(self.size)[indices]
        else:
            selected = np.asarray(list(indices), dtype=np.int64)
        return SplitData(
            name=name or self.name,
            ids=[self.ids[int(index)] for index in selected],
            features={m: np.asarray(self.features[m])[selected] for m in MODALITIES},
            masks={m: np.asarray(self.masks[m])[selected] for m in MODALITIES},
            classification=None if self.classification is None else self.classification[selected],
            regression=None if self.regression is None else self.regression[selected],
            annotations=[self.annotations[int(index)] for index in selected],
            mask_sources=self.mask_sources,
            metadata=self.metadata,
        )


@dataclass
class DatasetBundle:
    train: SplitData
    valid: SplitData
    test: SplitData
    source_path: Path
    label_path: Path | None
    label_mapping: dict[int, str]
    metadata: dict[str, Any]


@dataclass
class NormalizationStats:
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    epsilon: float = 1e-6
    feature_version: str = "aligned_50"
    fitted_sample_count: int = 0
    fitted_position_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.fitted_position_counts = dict(self.fitted_position_counts or {})
        for modality in MODALITIES:
            self.means[modality] = np.asarray(self.means[modality], dtype=np.float32)
            self.stds[modality] = np.asarray(self.stds[modality], dtype=np.float32)
            expected = (INPUT_DIMS[modality],)
            if self.means[modality].shape != expected or self.stds[modality].shape != expected:
                raise ValueError(f"Invalid normalization shape for {modality}")

    def transform(self, split: SplitData, name: str | None = None) -> SplitData:
        transformed: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            values = np.asarray(split.features[modality], dtype=np.float32)
            mean = self.means[modality].reshape(1, 1, -1)
            std = self.stds[modality].reshape(1, 1, -1)
            output = (values - mean) / std
            mask = np.asarray(split.masks[modality], dtype=bool)
            output[~mask] = 0.0
            transformed[modality] = np.nan_to_num(
                output, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        return replace(split, name=name or split.name, features=transformed)


def _as_ids(value: Any, n: int) -> list[str]:
    if value is None:
        return [str(index) for index in range(n)]
    values = list(np.asarray(value).reshape(-1))
    if len(values) != n:
        raise ValueError(f"ID count {len(values)} does not match sample count {n}")
    return [str(item) for item in values]


def _normalise_annotation(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    aliases = {
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "positive": "Positive",
        "pos": "Positive",
    }
    return aliases.get(lowered, text)


def _normalise_excel_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _make_sample_id(video_id: Any, clip_id: Any) -> str:
    return f"{_normalise_excel_scalar(video_id)}$_${_normalise_excel_scalar(clip_id)}"


def _read_label_table_with_openpyxl(path: Path) -> list[dict[str, str]]:
    import openpyxl  # type: ignore

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    rows = worksheet.iter_rows(values_only=True)
    header = [_normalise_excel_scalar(value) for value in next(rows)]
    records: list[dict[str, str]] = []
    for row in rows:
        record = {header[index]: _normalise_excel_scalar(value) for index, value in enumerate(row) if index < len(header)}
        records.append(record)
    return records


def _read_label_table_from_xlsx_xml(path: Path) -> list[dict[str, str]]:
    """Small stdlib fallback so schema validation also works without openpyxl."""

    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", _XML_NS):
                shared.append("".join(text.text or "" for text in item.findall(".//x:t", _XML_NS)))
        worksheet_name = next(
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        root = ET.fromstring(archive.read(worksheet_name))
        rows: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", _XML_NS):
            values: list[str] = []
            for cell in row.findall("x:c", _XML_NS):
                value = cell.find("x:v", _XML_NS)
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                values.append(_normalise_excel_scalar(text))
            rows.append(values)
    if not rows:
        return []
    header = rows[0]
    return [
        {header[index]: value for index, value in enumerate(row) if index < len(header)}
        for row in rows[1:]
    ]


def read_label_table(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    try:
        records = _read_label_table_with_openpyxl(path)
    except Exception:
        records = _read_label_table_from_xlsx_xml(path)
    result: dict[str, dict[str, str]] = {}
    for record in records:
        sample_id = record.get("id", "")
        if not sample_id:
            video_id = record.get("video_id", record.get("video", ""))
            clip_id = record.get("clip_id", record.get("clip", ""))
            if video_id or clip_id:
                sample_id = _make_sample_id(video_id, clip_id)
        if sample_id:
            result[sample_id] = record
    return result


def _find_explicit_mask(raw: Mapping[str, Any], modality: str, n: int, length: int) -> tuple[np.ndarray | None, str]:
    containers = [raw.get("masks"), raw.get("mask"), raw.get("valid_masks"), raw.get("availability_masks")]
    for container in containers:
        if isinstance(container, Mapping) and modality in container:
            candidate = np.asarray(container[modality]).astype(bool)
            if candidate.shape != (n, length):
                raise ValueError(f"Explicit {modality} mask shape {candidate.shape} is invalid")
            return candidate, "explicit_mask"
        if container is not None and not isinstance(container, Mapping):
            candidate_container = np.asarray(container)
            if candidate_container.shape == (n, len(MODALITIES), length):
                candidate = candidate_container[:, MODALITIES.index(modality), :].astype(bool)
                return candidate, "explicit_mask"
    for key in (f"{modality}_mask", f"{modality}_valid_mask"):
        if key in raw:
            candidate = np.asarray(raw[key]).astype(bool)
            if candidate.shape != (n, length):
                raise ValueError(f"Explicit {key} shape {candidate.shape} is invalid")
            return candidate, "explicit_mask"
    for key in (f"{modality}_lengths", f"{modality}_valid_lengths"):
        if key in raw:
            lengths = np.asarray(raw[key]).reshape(-1)
            if lengths.shape != (n,):
                raise ValueError(f"Explicit {key} shape {lengths.shape} is invalid")
            positions = np.arange(length)[None, :]
            return positions < np.clip(lengths[:, None], 0, length), "explicit_length"
    return None, "zero_row_heuristic"


def _row_nonzero_mask(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=-1)
    nonzero = np.any(np.abs(values) > 0.0, axis=-1)
    return finite & nonzero


def _extract_feature(raw: Mapping[str, Any], modality: str, n: int, length: int) -> tuple[np.ndarray, np.ndarray, str]:
    values = raw.get(modality)
    if values is None:
        # text_bert is deliberately not converted into a continuous text feature.
        output = np.zeros((n, length, INPUT_DIMS[modality]), dtype=np.float32)
        return output, np.zeros((n, length), dtype=bool), "missing_field"
    output = np.asarray(values, dtype=np.float32)
    expected = (n, length, INPUT_DIMS[modality])
    if output.shape != expected:
        raise ValueError(f"{modality} shape {output.shape} != {expected}")
    if not np.isfinite(output).all():
        raise ValueError(f"{modality} contains NaN or infinity")
    explicit, source = _find_explicit_mask(raw, modality, n, length)
    mask = _row_nonzero_mask(output) if explicit is None else explicit
    # Even a length mask cannot establish that an all-zero vector is observed.
    if source == "explicit_length":
        mask &= _row_nonzero_mask(output)
    output = output.copy()
    output[~mask] = 0.0
    return output, mask, source


def _annotation_and_label_maps(
    splits: Mapping[str, Mapping[str, Any]], label_table: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, str], dict[int, str], dict[str, Any]]:
    annotation_by_id: dict[str, str] = {}
    observations: dict[int, Counter[str]] = defaultdict(Counter)
    for raw in splits.values():
        ids = _as_ids(raw.get("id"), len(np.asarray(raw.get("classification_labels", []))))
        classes = np.asarray(raw.get("classification_labels", []), dtype=np.float64).reshape(-1)
        for sample_id, value in zip(ids, classes):
            record = label_table.get(sample_id, {})
            annotation = _normalise_annotation(record.get("annotation", record.get("label_name")))
            if annotation is not None:
                annotation_by_id[sample_id] = annotation
                if abs(value - round(value)) < 1e-6:
                    observations[int(round(value))][annotation] += 1
    mapping: dict[int, str] = {index: name for index, name in enumerate(CLASS_NAMES)}
    for class_id, counter in observations.items():
        if counter:
            mapping[class_id] = counter.most_common(1)[0][0]
    metadata = {
        "annotation_source": "label.xlsx" if label_table else "unavailable_default_class_mapping",
        "annotation_count": len(annotation_by_id),
        "annotation_class_observations": {
            str(class_id): dict(counter) for class_id, counter in observations.items()
        },
    }
    # The provided labels must not silently map one integer class to two names.
    reverse: dict[str, int] = {}
    for class_id, name in mapping.items():
        if name in reverse and reverse[name] != class_id:
            raise ValueError(f"Ambiguous annotation mapping for {name}")
        reverse[name] = class_id
    return annotation_by_id, mapping, metadata


def _extract_split(
    name: str,
    raw: Mapping[str, Any],
    annotation_by_id: Mapping[str, str],
    max_length: int = MAX_LENGTH,
) -> SplitData:
    classes = raw.get("classification_labels")
    n = len(np.asarray(classes).reshape(-1)) if classes is not None else None
    if n is None:
        for modality in MODALITIES:
            if modality in raw:
                n = int(np.asarray(raw[modality]).shape[0])
                break
    if n is None:
        raise ValueError(f"Cannot infer sample count for split {name}")
    if max_length != MAX_LENGTH:
        raise ValueError("Question 2 aligned interface is fixed at 50 positions")
    ids = _as_ids(raw.get("id"), n)
    features: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    for modality in MODALITIES:
        values, mask, source = _extract_feature(raw, modality, n, max_length)
        features[modality] = values
        masks[modality] = mask
        sources[modality] = source
    classification = None if classes is None else np.rint(np.asarray(classes, dtype=np.float64)).astype(np.int64)
    regression_value = raw.get("regression_labels")
    regression = None if regression_value is None else np.asarray(regression_value, dtype=np.float32).reshape(-1)
    annotations = [annotation_by_id.get(sample_id) for sample_id in ids]
    metadata = {
        "raw_keys": sorted(str(key) for key in raw.keys()),
        "text_bert_ignored": "text_bert" in raw and "text" not in raw,
        "zero_row_mask_ambiguity": any(source == "zero_row_heuristic" for source in sources.values()),
    }
    return SplitData(
        name=name,
        ids=ids,
        features=features,
        masks=masks,
        classification=classification,
        regression=regression,
        annotations=annotations,
        mask_sources=sources,
        metadata=metadata,
    )


def validate_split_partitions(splits: Mapping[str, SplitData]) -> dict[str, Any]:
    ids = {name: set(split.ids) for name, split in splits.items()}
    intersections: dict[str, list[str]] = {}
    names = list(ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(ids[left] & ids[right])
            intersections[f"{left}__{right}"] = overlap
    if any(intersections.values()):
        # The supplied split is retained, but the leakage is explicit in metadata.
        warning = "partition ID overlap detected"
    else:
        warning = ""
    return {
        "sample_counts": {name: len(value) for name, value in ids.items()},
        "intersections": {key: values[:20] for key, values in intersections.items()},
        "intersection_counts": {key: len(values) for key, values in intersections.items()},
        "warning": warning,
    }


def load_aligned_dataset(
    feature_path: str | Path,
    label_path: str | Path | None = None,
    max_length: int = MAX_LENGTH,
    limit: int | None = None,
) -> DatasetBundle:
    feature_path = Path(feature_path).resolve()
    label_path_obj = None if label_path is None else Path(label_path).resolve()
    with feature_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Aligned feature file must contain a mapping: {feature_path}")
    missing_splits = [name for name in ("train", "valid", "test") if name not in payload]
    if missing_splits:
        raise ValueError(f"Aligned feature file is missing splits: {missing_splits}")
    label_table = read_label_table(label_path_obj)
    annotation_by_id, mapping, label_metadata = _annotation_and_label_maps(
        {name: payload[name] for name in ("train", "valid", "test")}, label_table
    )
    splits = {
        name: _extract_split(name, payload[name], annotation_by_id, max_length=max_length)
        for name in ("train", "valid", "test")
    }
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be positive")
        splits = {name: split.subset(slice(0, limit), name=name) for name, split in splits.items()}
    partition_metadata = validate_split_partitions(splits)
    # Verify the documented neutral convention when annotations are available.
    neutral_zero_check: dict[str, Any] = {}
    for name, split in splits.items():
        if split.has_labels:
            neutral = np.asarray([item == "Neutral" for item in split.annotations], dtype=bool)
            neutral_zero_check[name] = {
                "annotation_neutral_count": int(neutral.sum()),
                "neutral_nonzero_intensity_count": int(
                    np.sum(neutral & (np.abs(split.regression) > 1e-6))
                ),
            }
    metadata = {
        "feature_version": "aligned_50",
        "feature_path": str(feature_path),
        "label_path": None if label_path_obj is None else str(label_path_obj),
        "partition": partition_metadata,
        "labels": label_metadata,
        "neutral_zero_check": neutral_zero_check,
        "mask_sources": {name: split.mask_sources for name, split in splits.items()},
        "mask_ambiguity_note": (
            "No explicit aligned masks were present; zero rows were treated as unavailable. "
            "Padding and physical missingness cannot be separated for those rows."
        ),
    }
    return DatasetBundle(
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
        source_path=feature_path,
        label_path=label_path_obj,
        label_mapping=mapping,
        metadata=metadata,
    )


def fit_normalization(split: SplitData, epsilon: float = 1e-6, feature_version: str = "aligned_50") -> NormalizationStats:
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    position_counts: dict[str, int] = {}
    for modality in MODALITIES:
        values = np.asarray(split.features[modality], dtype=np.float32)
        mask = np.asarray(split.masks[modality], dtype=bool)
        selected = values[mask]
        position_counts[modality] = int(mask.sum())
        if selected.size == 0:
            mean = np.zeros(INPUT_DIMS[modality], dtype=np.float32)
            std = np.ones(INPUT_DIMS[modality], dtype=np.float32)
        else:
            mean = np.nanmean(selected, axis=0).astype(np.float32)
            std = np.nanstd(selected, axis=0).astype(np.float32)
            std = np.where(np.isfinite(std) & (std > epsilon), std, 1.0).astype(np.float32)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        means[modality] = mean
        stds[modality] = np.maximum(std, float(epsilon)).astype(np.float32)
    return NormalizationStats(
        means=means,
        stds=stds,
        epsilon=epsilon,
        feature_version=feature_version,
        fitted_sample_count=split.size,
        fitted_position_counts=position_counts,
    )


def save_normalization(stats: NormalizationStats, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "epsilon": stats.epsilon,
        "feature_version": stats.feature_version,
        "fitted_sample_count": stats.fitted_sample_count,
        "fitted_position_counts": stats.fitted_position_counts,
    }
    np.savez(
        output,
        text_mean=stats.means["text"],
        text_std=stats.stds["text"],
        audio_mean=stats.means["audio"],
        audio_std=stats.stds["audio"],
        vision_mean=stats.means["vision"],
        vision_std=stats.stds["vision"],
        metadata=np.asarray([str(metadata)], dtype=object),
    )


def load_normalization(path: str | Path) -> NormalizationStats:
    with np.load(Path(path), allow_pickle=True) as payload:
        metadata_text = str(payload["metadata"].reshape(-1)[0]) if "metadata" in payload else "{}"
        try:
            metadata = ast.literal_eval(metadata_text)
        except Exception:
            metadata = {}
        return NormalizationStats(
            means={m: np.asarray(payload[f"{m}_mean"], dtype=np.float32) for m in MODALITIES},
            stds={m: np.asarray(payload[f"{m}_std"], dtype=np.float32) for m in MODALITIES},
            epsilon=float(metadata.get("epsilon", 1e-6)),
            feature_version=str(metadata.get("feature_version", "aligned_50")),
            fitted_sample_count=int(metadata.get("fitted_sample_count", 0)),
            fitted_position_counts=dict(metadata.get("fitted_position_counts", {})),
        )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()
