from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np
import torch

from . import MODALITIES
from .data import SplitData
from .masking import MaskScenario, mask_statistics, unstack_masks
from .metrics import aggregate_scores, composite_score, confusion_rows, evaluate_predictions
from .torchdata import make_loader


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def predict_split(
    model: torch.nn.Module,
    split: SplitData,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    masks_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if masks_override is not None:
        split = replace(split, masks=unstack_masks(masks_override))
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    loader = make_loader(split, batch_size, shuffle=False, num_workers=num_workers, require_labels=False)
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    intensity: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    all_unavailable: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = _move_nested(batch["features"], device_obj)
            masks = _move_nested(batch["masks"], device_obj)
            features = {
                modality: features[modality].masked_fill(
                    ~masks[modality].unsqueeze(-1), 0.0
                )
                for modality in MODALITIES
            }
            output = model(features, masks)
            logits.append(output["logits"].detach().cpu().numpy())
            probabilities.append(output["probabilities"].detach().cpu().numpy())
            intensity.append(output["intensity"].detach().cpu().numpy())
            gate_weights.append(output["gate_weights"].detach().cpu().numpy())
            coverage.append(output["coverage"].detach().cpu().numpy())
            all_unavailable.append(output["all_unavailable"].detach().cpu().numpy())
    if not logits:
        return {
            "logits": np.zeros((0, 3), dtype=np.float32),
            "probabilities": np.zeros((0, 3), dtype=np.float32),
            "intensity": np.zeros((0,), dtype=np.float32),
            "predicted_class": np.zeros((0,), dtype=np.int64),
            "gate_weights": np.zeros((0, 50, 3), dtype=np.float32),
            "coverage": np.zeros((0, 3), dtype=np.float32),
            "all_unavailable": np.zeros((0,), dtype=bool),
        }
    result = {
        "logits": np.concatenate(logits),
        "probabilities": np.concatenate(probabilities),
        "intensity": np.concatenate(intensity),
        "gate_weights": np.concatenate(gate_weights),
        "coverage": np.concatenate(coverage),
        "all_unavailable": np.concatenate(all_unavailable).astype(bool),
    }
    result["predicted_class"] = np.argmax(result["probabilities"], axis=1).astype(np.int64)
    return result


def evaluate_split(
    model: torch.nn.Module,
    split: SplitData,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    masks_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not split.has_labels:
        raise ValueError(f"Split {split.name} has no labels")
    predictions = predict_split(
        model, split, device=device, batch_size=batch_size, masks_override=masks_override
    )
    metrics = evaluate_predictions(
        split.classification,
        predictions["predicted_class"],
        split.regression,
        predictions["intensity"],
    )
    metrics["all_unavailable_count"] = int(predictions["all_unavailable"].sum())
    return metrics, predictions


def evaluate_scenarios(
    model: torch.nn.Module,
    split: SplitData,
    scenarios: Iterable[MaskScenario],
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    base_masks = split.mask_array()
    for scenario in scenarios:
        metrics, predictions = evaluate_split(
            model,
            split,
            device=device,
            batch_size=batch_size,
            masks_override=scenario.masks,
        )
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "missing_modalities": "+".join(scenario.missing_modalities),
            "missing_fraction": scenario.fraction,
            "missing_position": scenario.position,
            "relation": scenario.relation,
            "mask_seed": scenario.seed,
            **metrics,
            "composite_score": composite_score(metrics),
        }
        row["mask_statistics"] = mask_statistics(base_masks, scenario.masks)
        rows.append(row)
        confusion.extend(confusion_rows(metrics["confusion_matrix"], scenario=scenario.name))
        row["gate_weight_mean"] = np.asarray(predictions["gate_weights"]).mean(axis=(0, 1)).tolist()
    return rows, confusion


def selection_score(metric_rows: Iterable[dict[str, Any]]) -> float:
    rows = list(metric_rows)
    values = [composite_score(row) for row in rows]
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if values:
        return float(np.mean(values))
    # Pearson is undefined for a constant prediction; retain that fact in the
    # reports, but use the other terms only as an early-stopping fallback.
    fallback = []
    for row in rows:
        fallback.append(
            (
                float(row.get("accuracy", 0.0))
                + float(row.get("macro_f1", 0.0))
                + (1.0 - float(row.get("mae", 6.0)) / 6.0)
            )
            / 3.0
        )
    return float(np.mean(fallback)) if fallback else float("-inf")


def classification_priority_score(metric_rows: Iterable[dict[str, Any]]) -> float:
    """Predeclared V3 development score emphasizing balanced classification.

    Regression metrics remain reported and constrained separately, while this
    score prevents MAE/Pearson from hiding a collapse of the Neutral class.
    """

    rows = [dict(row) for row in metric_rows]
    if not rows:
        return float("-inf")
    macro = np.asarray([float(row.get("macro_f1", 0.0)) for row in rows], dtype=np.float64)
    neutral = np.asarray([float(row.get("f1_neutral", 0.0)) for row in rows], dtype=np.float64)
    accuracy = np.asarray([float(row.get("accuracy", 0.0)) for row in rows], dtype=np.float64)
    values = (
        0.50 * float(np.mean(macro))
        + 0.20 * float(np.mean(neutral))
        + 0.15 * float(np.mean(accuracy))
        + 0.15 * float(np.min(macro))
    )
    return float(values)


def majority_and_mean_baseline(split: SplitData) -> dict[str, Any]:
    if not split.has_labels or split.size == 0:
        raise ValueError("Baseline requires a non-empty labelled split")
    counts = np.bincount(split.classification, minlength=3)
    majority = int(np.argmax(counts))
    predictions_class = np.full(split.size, majority, dtype=np.int64)
    predictions_regression = np.full(split.size, float(np.mean(split.regression)), dtype=np.float32)
    metrics = evaluate_predictions(
        split.classification,
        predictions_class,
        split.regression,
        predictions_regression,
    )
    metrics.update({"baseline": "majority_class_and_train_split_mean_regression", "majority_class": majority})
    return metrics


def aggregate_scenario_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    clean_rows = []
    for row in rows:
        copy = dict(row)
        copy.pop("mask_statistics", None)
        copy.pop("gate_weight_mean", None)
        clean_rows.append(copy)
    return aggregate_scores(clean_rows)
