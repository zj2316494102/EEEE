from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")

COMMON_WORD_EQUIVALENTS = {
    "cant": "cannot",
    "cannot": "cannot",
    "wont": "willnot",
    "willnot": "willnot",
    "dont": "donot",
    "donot": "donot",
    "doesnt": "doesnot",
    "doesnot": "doesnot",
    "didnt": "didnot",
    "didnot": "didnot",
    "isnt": "isnot",
    "isnot": "isnot",
    "arent": "arenot",
    "arenot": "arenot",
    "wasnt": "wasnot",
    "werent": "werenot",
    "im": "iam",
    "iam": "iam",
    "ive": "ihave",
    "ihave": "ihave",
    "ill": "iwill",
    "iwill": "iwill",
    "id": "iwould",
    "iwould": "iwould",
    "youre": "youare",
    "youare": "youare",
    "theyre": "theyare",
    "theyare": "theyare",
    "weve": "wehave",
    "wehave": "wehave",
    "theyve": "theyhave",
    "theyhave": "theyhave",
    "thats": "thatis",
    "thatis": "thatis",
}


@dataclass
class TextToken:
    raw: str
    normalized: str
    start_char: int
    end_char: int


def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    value = value.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9']+", "", value)
    return value


def normalize_match_key(value: str) -> str:
    """Return a conservative key used only for matching, never for output text."""
    normalized = normalize_token(value).replace("'", "")
    return COMMON_WORD_EQUIVALENTS.get(normalized, normalized)


def tokenize_text(text: str) -> list[TextToken]:
    result: list[TextToken] = []
    for match in TOKEN_RE.finditer(unicodedata.normalize("NFKC", text or "")):
        raw = match.group(0)
        normalized = normalize_token(raw)
        if normalized:
            result.append(TextToken(raw, normalized, match.start(), match.end()))
    return result


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def _similar_enough(left: str, right: str, max_ratio: float) -> bool:
    if not left or not right:
        return False
    distance = _levenshtein(left, right)
    denominator = max(len(left), len(right), 1)
    return distance / denominator <= max_ratio


def _match_candidate(
    target_keys: list[str],
    source_keys: list[str],
    target_start: int,
    target_count: int,
    source_start: int,
    source_count: int,
    max_edit_ratio: float,
) -> tuple[str, float] | None:
    target_key = "".join(target_keys[target_start : target_start + target_count])
    source_key = "".join(source_keys[source_start : source_start + source_count])
    if not target_key or not source_key:
        return None
    if target_key == source_key:
        if target_count == 1 and source_count == 1:
            return "exact", 3.0
        return "equivalent_compound", 2.65
    if target_count == 1 and source_count == 1 and _similar_enough(target_key, source_key, max_edit_ratio):
        ratio = _levenshtein(target_key, source_key) / max(len(target_key), len(source_key), 1)
        return "edit_distance", 1.25 - ratio
    return None


def _monotonic_alignment_pairs(
    target_keys: list[str],
    source_keys: list[str],
    max_edit_ratio: float,
) -> list[tuple[list[int], list[int], str]]:
    """Align two short word sequences with monotonic DP and auditable operations."""
    target_count = len(target_keys)
    source_count = len(source_keys)
    scores = np.full((target_count + 1, source_count + 1), -np.inf, dtype=np.float64)
    operations: dict[tuple[int, int], tuple[int, int, str, str]] = {}
    scores[0, 0] = 0.0

    def update(
        next_i: int,
        next_j: int,
        score: float,
        previous_i: int,
        previous_j: int,
        operation: str,
        match_type: str = "",
    ) -> None:
        if score > scores[next_i, next_j] + 1e-9:
            scores[next_i, next_j] = score
            operations[(next_i, next_j)] = (previous_i, previous_j, operation, match_type)

    for i in range(target_count + 1):
        for j in range(source_count + 1):
            if not np.isfinite(scores[i, j]):
                continue
            if i < target_count:
                update(i + 1, j, scores[i, j] - 0.75, i, j, "skip_target")
            if j < source_count:
                update(i, j + 1, scores[i, j] - 0.15, i, j, "skip_source")
            for target_span, source_span in ((1, 1), (1, 2), (2, 1)):
                if i + target_span > target_count or j + source_span > source_count:
                    continue
                candidate = _match_candidate(
                    target_keys,
                    source_keys,
                    i,
                    target_span,
                    j,
                    source_span,
                    max_edit_ratio,
                )
                if candidate is None:
                    continue
                match_type, match_score = candidate
                update(i + target_span, j + source_span, scores[i, j] + match_score, i, j, "match", match_type)

    pairs: list[tuple[list[int], list[int], str]] = []
    i, j = target_count, source_count
    while i > 0 or j > 0:
        previous = operations.get((i, j))
        if previous is None:
            if i > 0:
                i -= 1
            elif j > 0:
                j -= 1
            continue
        previous_i, previous_j, operation, match_type = previous
        if operation == "match":
            pairs.append((list(range(previous_i, i)), list(range(previous_j, j)), match_type))
        i, j = previous_i, previous_j
    pairs.reverse()
    return pairs


def _clip_interval(start: float, end: float, duration: float) -> tuple[float, float]:
    duration = max(float(duration), 0.0)
    start = float(np.clip(start, 0.0, duration))
    end = float(np.clip(end, 0.0, duration))
    if end < start:
        start, end = end, start
    return start, end


def _float32_interval(
    start: float,
    end: float,
    duration: float,
    minimum_duration: float = 0.0,
) -> tuple[float, float] | None:
    """Validate an interval after the precision used by persisted audit data."""
    start32 = float(np.float32(start))
    end32 = float(np.float32(end))
    duration32 = float(np.float32(max(duration, 0.0)))
    if not (np.isfinite(start32) and np.isfinite(end32)):
        return None
    if start32 < 0.0 or end32 > duration32 or end32 <= start32:
        return None
    if end32 - start32 < float(minimum_duration):
        return None
    return start32, end32


def _fallback_interval(
    target_index: int,
    unmatched_indices: list[int],
    records: list[dict[str, Any]],
    duration: float,
    minimum_duration: float,
) -> tuple[float, float] | None:
    previous = next(
        (records[i] for i in range(target_index - 1, -1, -1) if records[i].get("matched") and records[i].get("time_valid")),
        None,
    )
    following = next(
        (records[i] for i in range(target_index + 1, len(records)) if records[i].get("matched") and records[i].get("time_valid")),
        None,
    )
    group_position = unmatched_indices.index(target_index)
    group_size = len(unmatched_indices)
    if previous and following:
        left = float(previous["end"])
        right = float(following["start"])
    elif previous:
        left = float(previous["end"])
        right = duration
    elif following:
        left = 0.0
        right = float(following["start"])
    else:
        left, right = 0.0, duration
    if right < left:
        right = left
    start = left + (right - left) * group_position / group_size
    end = left + (right - left) * (group_position + 1) / group_size
    start, end = _clip_interval(start, end, duration)
    return _float32_interval(start, end, duration, minimum_duration=minimum_duration)


def align_text_words(
    raw_text: str,
    asr_words: Iterable[dict[str, Any]],
    duration: float,
    max_edit_ratio: float = 0.34,
    fallback_confidence: float = 0.20,
    fallback_min_interval_s: float = 0.01,
) -> list[dict[str, Any]]:
    """Align provided-text tokens to ASR timestamps without replacing the text.

    The sequence matcher provides deterministic exact/near matching. Unmatched
    source words are never silently discarded; target words receive a fixed
    boundary or uniform-span fallback and are marked low confidence.
    """

    target = tokenize_text(raw_text)
    source = []
    for index, item in enumerate(asr_words):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        start, end = _clip_interval(float(start), float(end), duration)
        if end <= start:
            continue
        text = str(item.get("text", item.get("word", ""))).strip()
        normalized = normalize_token(text)
        if not normalized:
            continue
        source.append(
            {
                "source_index": index,
                "raw": text,
                "normalized": normalized,
                "match_key": normalize_match_key(normalized),
                "start": start,
                "end": end,
                "confidence": float(item.get("confidence", 1.0) or 1.0),
            }
        )

    target_tokens = [item.normalized for item in target]
    target_keys = [normalize_match_key(item) for item in target_tokens]
    source_tokens = [item["normalized"] for item in source]
    source_keys = [item["match_key"] for item in source]
    records: list[dict[str, Any]] = [
        {
            "word": item.raw,
            "normalized": item.normalized,
            "start_char": item.start_char,
            "end_char": item.end_char,
            "start": None,
            "end": None,
            "confidence": 0.0,
            "match_type": "unmatched",
            "fallback": True,
            "source_indices": [],
            "matched": False,
            "time_valid": False,
            "timestamp_source": "unassigned",
            "invalid_reason": "not_aligned",
        }
        for item in target
    ]

    for target_indices, source_indices, match_type in _monotonic_alignment_pairs(
        target_keys, source_keys, max_edit_ratio
    ):
        source_items = [source[index] for index in source_indices]
        start = min(item["start"] for item in source_items)
        end = max(item["end"] for item in source_items)
        valid_interval = _float32_interval(start, end, duration)
        base_confidence = min(float(np.clip(item.get("confidence", 1.0), 0.0, 1.0)) for item in source_items)
        if match_type == "exact":
            confidence = max(base_confidence, 1.0)
            timestamp_source = "asr_exact"
        elif match_type == "equivalent_compound":
            confidence = min(base_confidence, 0.85)
            timestamp_source = "asr_equivalent"
        else:
            confidence = min(base_confidence, 0.65)
            timestamp_source = "asr_edit"
        for target_index in target_indices:
            current = records[target_index]
            current.update(
                {
                    "start": valid_interval[0] if valid_interval is not None else None,
                    "end": valid_interval[1] if valid_interval is not None else None,
                    "confidence": confidence,
                    "match_type": match_type,
                    "fallback": False,
                    "source_indices": [item["source_index"] for item in source_items],
                    "matched": valid_interval is not None,
                    "time_valid": valid_interval is not None,
                    "timestamp_source": timestamp_source,
                    "invalid_reason": "" if valid_interval is not None else "non_positive_asr_interval",
                }
            )

    unmatched = [index for index, record in enumerate(records) if not record["matched"]]
    for index in unmatched:
        fallback = _fallback_interval(index, unmatched, records, duration, float(fallback_min_interval_s))
        if fallback is None:
            records[index].update(
                {
                    "start": None,
                    "end": None,
                    "confidence": fallback_confidence,
                    "match_type": "invalid_fallback",
                    "fallback": True,
                    "time_valid": False,
                    "timestamp_source": "invalid",
                    "invalid_reason": "fallback_interval_below_0.01s_or_float32_collapse",
                }
            )
            continue
        start, end = fallback
        if end - start < float(fallback_min_interval_s):
            records[index].update(
                {
                    "start": None,
                    "end": None,
                    "confidence": fallback_confidence,
                    "match_type": "invalid_fallback",
                    "fallback": True,
                    "time_valid": False,
                    "timestamp_source": "invalid",
                    "invalid_reason": "fallback_interval_below_configured_minimum",
                }
            )
            continue
        records[index].update(
            {
                "start": start,
                "end": end,
                "confidence": fallback_confidence,
                "match_type": "boundary_or_uniform_fallback",
                "fallback": True,
                "time_valid": True,
                "timestamp_source": "fallback_boundary_or_uniform",
                "invalid_reason": "",
            }
        )
    for record in records:
        if record.get("time_valid") and record.get("start") is not None and record.get("end") is not None:
            persisted_interval = _float32_interval(record["start"], record["end"], duration)
            if persisted_interval is None:
                record.update(
                    {
                        "start": None,
                        "end": None,
                        "time_valid": False,
                        "timestamp_source": "invalid",
                        "invalid_reason": "float32_persistence_collapse",
                    }
                )
            else:
                record["start"], record["end"] = persisted_interval
    for record in records:
        record.pop("matched", None)
    return records


def make_time_grid(duration: float, length: int = 50) -> np.ndarray:
    duration = max(float(duration), 0.0)
    edges = np.linspace(0.0, duration, int(length) + 1, dtype=np.float64)
    return np.stack([edges[:-1], edges[1:]], axis=1)


def convolution_spans(
    feature_count: int,
    sample_rate: int,
    conv_kernel: Iterable[int],
    conv_stride: Iterable[int],
    duration: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    jump = 1
    receptive_field = 1
    for kernel, stride in zip(conv_kernel, conv_stride):
        receptive_field += (int(kernel) - 1) * jump
        jump *= int(stride)
    starts = np.arange(int(feature_count), dtype=np.float64) * jump
    ends = starts + receptive_field
    spans = np.stack([starts / sample_rate, ends / sample_rate], axis=1)
    spans[:, 0] = np.clip(spans[:, 0], 0.0, max(duration, 0.0))
    spans[:, 1] = np.clip(spans[:, 1], 0.0, max(duration, 0.0))
    valid = spans[:, 1] > spans[:, 0]
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    return spans, valid, {"effective_stride_samples": jump, "receptive_field_samples": receptive_field}


def frame_spans_from_centers(centers: np.ndarray, duration: float, nominal_fps: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    half = 0.5 / max(float(nominal_fps), 1e-6)
    spans = np.stack([centers - half, centers + half], axis=1)
    spans[:, 0] = np.clip(spans[:, 0], 0.0, max(float(duration), 0.0))
    spans[:, 1] = np.clip(spans[:, 1], 0.0, max(float(duration), 0.0))
    return spans


def overlap_pool(
    features: np.ndarray,
    spans: np.ndarray,
    grid: np.ndarray,
    quality: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    spans = np.asarray(spans, dtype=np.float64).reshape((-1, 2))
    grid = np.asarray(grid, dtype=np.float64).reshape((-1, 2))
    if features.ndim != 2 or features.shape[0] != spans.shape[0]:
        raise ValueError(f"Feature/span shape mismatch: {features.shape} vs {spans.shape}")
    quality_array = np.ones(features.shape[0], dtype=np.float64) if quality is None else np.asarray(quality, dtype=np.float64)
    valid_array = np.ones(features.shape[0], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    output = np.zeros((grid.shape[0], features.shape[1]), dtype=np.float32)
    masks = np.zeros(grid.shape[0], dtype=np.uint8)
    denominators = np.zeros(grid.shape[0], dtype=np.float32)
    for k, (grid_start, grid_end) in enumerate(grid):
        intersection_start = np.maximum(spans[:, 0], grid_start)
        intersection_end = np.minimum(spans[:, 1], grid_end)
        intersection = np.maximum(intersection_end - intersection_start, 0.0)
        span_length = np.maximum(spans[:, 1] - spans[:, 0], 1e-12)
        weights = quality_array * intersection / span_length
        weights[~valid_array] = 0.0
        weights[~np.isfinite(weights)] = 0.0
        denominator = float(weights.sum())
        denominators[k] = denominator
        if denominator > 0:
            output[k] = (features * weights[:, None]).sum(axis=0) / denominator
            masks[k] = 1
    return output, masks, denominators


def standardize_aligned(
    values: np.ndarray,
    masks: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if values.ndim != 3 or masks.shape != values.shape[:2]:
        raise ValueError(f"Standardization shape mismatch: values={values.shape}, masks={masks.shape}")
    flat = values.reshape((-1, values.shape[-1]))
    valid = masks.reshape(-1)
    if not valid.any():
        mean = np.zeros(values.shape[-1], dtype=np.float32)
        std = np.ones(values.shape[-1], dtype=np.float32)
    else:
        valid_values = flat[valid]
        mean = valid_values.mean(axis=0).astype(np.float32)
        std = valid_values.std(axis=0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > epsilon), std, 1.0).astype(np.float32)
    standardized = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    standardized[~masks] = 0.0
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return standardized, mean, std, valid.reshape(masks.shape)
