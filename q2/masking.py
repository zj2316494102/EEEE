from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import MODALITIES


@dataclass
class MaskScenario:
    name: str
    masks: np.ndarray  # [N, 3, T], bool
    missing_modalities: tuple[str, ...]
    fraction: float
    position: str
    relation: str
    seed: int | None = None


def stack_masks(masks: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(masks, dict):
        output = np.stack([np.asarray(masks[modality], dtype=bool) for modality in MODALITIES], axis=1)
    else:
        output = np.asarray(masks, dtype=bool)
    if output.ndim != 3 or output.shape[1] != len(MODALITIES):
        raise ValueError(f"Masks must have shape [N, 3, T], got {output.shape}")
    return output


def unstack_masks(masks: np.ndarray) -> dict[str, np.ndarray]:
    values = stack_masks(masks)
    return {modality: values[:, index].copy() for index, modality in enumerate(MODALITIES)}


def _choose_modality_count(rng: np.random.Generator, probabilities: Sequence[float]) -> int:
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
        weights = np.asarray([0.55, 0.35, 0.10], dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, 4), p=weights))


def _restore_one_available(mask: np.ndarray, original: np.ndarray, rng: np.random.Generator) -> None:
    if mask.any() or not original.any():
        return
    available = np.flatnonzero(original)
    mask[int(rng.choice(available))] = True


def _apply_random_block(
    mask: np.ndarray,
    rng: np.random.Generator,
    max_fraction: float,
    max_segments: int = 3,
) -> list[tuple[int, int]]:
    original = mask.copy()
    available = np.flatnonzero(original)
    if len(available) <= 1:
        return []
    max_fraction = float(np.clip(max_fraction, 1.0 / max(len(available), 1), 1.0))
    max_length = max(1, int(np.ceil(len(available) * max_fraction)))
    segment_count = int(rng.integers(1, max_segments + 1))
    intervals: list[tuple[int, int]] = []
    for _ in range(segment_count):
        length = int(rng.integers(1, max_length + 1))
        if len(available) - length <= 0:
            length = len(available) - 1
        if length <= 0:
            break
        source_start = int(rng.integers(0, max(1, len(available) - length + 1)))
        start = int(available[source_start])
        last = int(available[min(len(available) - 1, source_start + length - 1)])
        end = min(mask.shape[0], last + 1)
        mask[start:end] = False
        intervals.append((start, end))
    _restore_one_available(mask, original, rng)
    return intervals


def generate_contiguous_block_masks(
    base_masks: dict[str, np.ndarray] | np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.75,
    max_fraction: float = 0.60,
    modality_probabilities: Sequence[float] = (0.55, 0.35, 0.10),
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Generate online continuous-block missingness without mutating the source mask."""

    original = stack_masks(base_masks)
    output = original.copy()
    records: list[dict[str, object]] = []
    for sample_index in range(output.shape[0]):
        if float(rng.random()) >= float(probability):
            records.append({"sample_index": sample_index, "applied": False, "modalities": []})
            continue
        available_modalities = [index for index in range(3) if original[sample_index, index].any()]
        if not available_modalities:
            records.append({"sample_index": sample_index, "applied": False, "modalities": [], "all_unavailable": True})
            continue
        count = min(_choose_modality_count(rng, modality_probabilities), len(available_modalities))
        selected = sorted(int(item) for item in rng.choice(available_modalities, size=count, replace=False))
        intervals: dict[str, list[tuple[int, int]]] = {}
        for modality_index in selected:
            intervals[MODALITIES[modality_index]] = _apply_random_block(
                output[sample_index, modality_index], rng, max_fraction
            )
        if not output[sample_index].any():
            # Keep the last piece of information in the sample observable.
            candidates = np.argwhere(original[sample_index])
            if len(candidates):
                chosen = candidates[int(rng.integers(0, len(candidates)))]
                output[sample_index, int(chosen[0]), int(chosen[1])] = True
        records.append(
            {
                "sample_index": sample_index,
                "applied": True,
                "modalities": [MODALITIES[index] for index in selected],
                "intervals": intervals,
            }
        )
    return output, records


def apply_whole_modality_dropout(
    base_masks: dict[str, np.ndarray] | np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.0,
    keep_at_least_one_modality: bool = True,
) -> np.ndarray:
    """Drop one complete modality for selected samples.

    This is intentionally separate from contiguous time-block corruption. It
    models a sample-level unavailable modality while preserving at least one
    usable modality by default. The source mask is never mutated.
    """

    output = stack_masks(base_masks).copy()
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= 0.0:
        return output
    for sample_index in range(output.shape[0]):
        if float(rng.random()) >= probability:
            continue
        available = np.flatnonzero(output[sample_index].any(axis=1))
        if len(available) == 0 or (keep_at_least_one_modality and len(available) <= 1):
            continue
        chosen = int(rng.choice(available))
        candidate = output[sample_index].copy()
        candidate[chosen] = False
        if keep_at_least_one_modality and not candidate.any():
            continue
        output[sample_index] = candidate
    return output


def generate_random_point_masks(
    base_masks: dict[str, np.ndarray] | np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.75,
    missing_fraction: float = 0.20,
) -> np.ndarray:
    """A random-point counterpart used only for the prescribed ablation."""

    original = stack_masks(base_masks)
    output = original.copy()
    for sample_index in range(output.shape[0]):
        if float(rng.random()) >= probability:
            continue
        selected = [index for index in range(3) if original[sample_index, index].any()]
        if not selected:
            continue
        count = int(rng.integers(1, len(selected) + 1))
        selected = list(rng.choice(selected, size=count, replace=False))
        for modality_index in selected:
            available = np.flatnonzero(original[sample_index, modality_index])
            drop_count = min(len(available) - 1, max(1, int(np.ceil(len(available) * missing_fraction))))
            if drop_count > 0:
                dropped = rng.choice(available, size=drop_count, replace=False)
                output[sample_index, modality_index, dropped] = False
    return output


def _interval_for_position(mask: np.ndarray, fraction: float, position: str) -> tuple[int, int, int]:
    available = np.flatnonzero(mask)
    if len(available) <= 1 or fraction <= 0:
        return 0, 0, 0
    count = min(len(available) - 1, max(1, int(np.ceil(len(available) * fraction))))
    if position == "front":
        index = 0
    elif position == "back":
        index = len(available) - count
    elif position == "middle":
        index = max(0, (len(available) - count) // 2)
    else:
        raise ValueError(f"Unknown missing position: {position}")
    start = int(available[index])
    end = int(available[min(len(available) - 1, index + count - 1)]) + 1
    return start, min(mask.shape[0], end), count


def _all_subsets() -> tuple[tuple[int, ...], ...]:
    return (
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    )


def build_fixed_validation_scenarios(
    base_masks: dict[str, np.ndarray] | np.ndarray,
    fractions: Sequence[float] = (0.10, 0.20, 0.40, 0.60),
    positions: Sequence[str] = ("front", "middle", "back"),
    groups: Sequence[Sequence[int]] | None = None,
    relation_modes: Sequence[str] = ("overlap", "stagger"),
    seeds: Sequence[int] = (2026, 42, 3407),
) -> list[MaskScenario]:
    """Create a fixed, reproducible validation grid shared by all model variants."""

    original = stack_masks(base_masks)
    scenarios = [
        MaskScenario(
            name="complete",
            masks=original.copy(),
            missing_modalities=(),
            fraction=0.0,
            position="none",
            relation="none",
            seed=None,
        )
    ]
    groups = groups or _all_subsets()
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for fraction in fractions:
            for position in positions:
                for group in groups:
                    group_indices = tuple(int(index) for index in group)
                    relations = ("single",) if len(group_indices) == 1 else tuple(relation_modes)
                    for relation in relations:
                        masked = original.copy()
                        for offset, modality_index in enumerate(group_indices):
                            for sample_index in range(masked.shape[0]):
                                start, end, count = _interval_for_position(
                                    original[sample_index, modality_index], float(fraction), position
                                )
                                if relation == "stagger" and len(group_indices) > 1 and end > start:
                                    shift = int(rng.integers(0, max(1, masked.shape[2] // len(group_indices) + 1)))
                                    start = min(masked.shape[2], start + offset * shift)
                                    end = min(masked.shape[2], end + offset * shift)
                                if count > 0:
                                    masked[sample_index, modality_index, start:end] = False
                        # A scenario is not allowed to silently turn a sample into a fully blank input.
                        for sample_index in range(masked.shape[0]):
                            for modality_index in group_indices:
                                if original[sample_index, modality_index].any() and not masked[sample_index, modality_index].any():
                                    # Preserve at least one observed position in every
                                    # selected modality, even when the source mask is
                                    # sparse and a time interval covers several gaps.
                                    first_available = int(np.flatnonzero(original[sample_index, modality_index])[0])
                                    masked[sample_index, modality_index, first_available] = True
                            if not masked[sample_index].any() and original[sample_index].any():
                                candidate = np.argwhere(original[sample_index])
                                chosen = candidate[int(rng.integers(0, len(candidate)))]
                                masked[sample_index, int(chosen[0]), int(chosen[1])] = True
                        modality_name = "+".join(MODALITIES[index] for index in group_indices)
                        relation_name = relation if len(group_indices) > 1 else "single"
                        scenarios.append(
                            MaskScenario(
                                name=f"{modality_name}__{int(round(fraction * 100)):02d}__{position}__{relation_name}__s{seed}",
                                masks=masked,
                                missing_modalities=tuple(MODALITIES[index] for index in group_indices),
                                fraction=float(fraction),
                                position=position,
                                relation=relation_name,
                                seed=int(seed),
                            )
                        )
    return scenarios


def _contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return runs


def mask_statistics(base_masks: dict[str, np.ndarray] | np.ndarray, masks: dict[str, np.ndarray] | np.ndarray) -> dict[str, object]:
    original = stack_masks(base_masks)
    current = stack_masks(masks)
    if original.shape != current.shape:
        raise ValueError(f"Base/current mask shape mismatch: {original.shape} vs {current.shape}")
    result: dict[str, object] = {"sample_count": int(original.shape[0]), "modalities": {}}
    modality_stats: dict[str, object] = {}
    for modality_index, modality in enumerate(MODALITIES):
        original_count = original[:, modality_index].sum(axis=1)
        observed_count = (original[:, modality_index] & current[:, modality_index]).sum(axis=1)
        missing = original[:, modality_index] & ~current[:, modality_index]
        ratios = np.divide(
            original_count - observed_count,
            np.maximum(original_count, 1),
            where=original_count > 0,
            out=np.zeros_like(original_count, dtype=np.float64),
        )
        longest = np.asarray(
            [max((end - start for start, end in _contiguous_runs(row)), default=0) for row in missing],
            dtype=np.float64,
        )
        modality_stats[modality] = {
            "original_positions": int(original_count.sum()),
            "observed_positions": int(observed_count.sum()),
            "missing_fraction_mean": float(ratios.mean()) if len(ratios) else 0.0,
            "missing_fraction_max": float(ratios.max()) if len(ratios) else 0.0,
            "longest_missing_run_mean_over_50": float((longest / max(original.shape[2], 1)).mean()) if len(longest) else 0.0,
            "longest_missing_run_max_over_50": float((longest / max(original.shape[2], 1)).max()) if len(longest) else 0.0,
        }
    result["modalities"] = modality_stats
    return result
