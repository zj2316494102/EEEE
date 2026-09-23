from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .data import CLASS_NAMES


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], class_count: int = 3) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        true_index, pred_index = int(true), int(pred)
        if 0 <= true_index < class_count and 0 <= pred_index < class_count:
            matrix[true_index, pred_index] += 1
    return matrix


def _f1_per_class(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )


def pearson_correlation(y_true: Iterable[float], y_pred: Iterable[float]) -> float | None:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[valid], pred[valid]
    if len(true) < 2:
        return None
    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denominator = float(np.sqrt(np.sum(true_centered**2) * np.sum(pred_centered**2)))
    if denominator <= 0:
        return None
    return float(np.sum(true_centered * pred_centered) / denominator)


def evaluate_predictions(
    y_true_class: Iterable[int],
    y_pred_class: Iterable[int],
    y_true_regression: Iterable[float],
    y_pred_regression: Iterable[float],
) -> dict[str, Any]:
    true_class = np.asarray(list(y_true_class), dtype=np.int64)
    pred_class = np.asarray(list(y_pred_class), dtype=np.int64)
    true_regression = np.asarray(list(y_true_regression), dtype=np.float64)
    pred_regression = np.asarray(list(y_pred_regression), dtype=np.float64)
    if not (len(true_class) == len(pred_class) == len(true_regression) == len(pred_regression)):
        raise ValueError("Metric arrays must have the same length")
    matrix = confusion_matrix(true_class, pred_class, class_count=len(CLASS_NAMES))
    f1 = _f1_per_class(matrix)
    support = matrix.sum(axis=1).astype(np.float64)
    total = max(float(support.sum()), 1.0)
    metrics: dict[str, Any] = {
        "sample_count": int(len(true_class)),
        "accuracy": float(np.mean(true_class == pred_class)) if len(true_class) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "weighted_f1": float(np.sum(f1 * support) / total),
        "mae": float(np.mean(np.abs(true_regression - pred_regression))) if len(true_regression) else 0.0,
        "pearson": pearson_correlation(true_regression, pred_regression),
        "f1_negative": float(f1[0]),
        "f1_neutral": float(f1[1]),
        "f1_positive": float(f1[2]),
        "support_negative": int(support[0]),
        "support_neutral": int(support[1]),
        "support_positive": int(support[2]),
        "confusion_matrix": matrix.tolist(),
    }
    return metrics


def composite_score(metrics: dict[str, Any]) -> float | None:
    """The predeclared four-term development score from the solution document."""

    pearson = _as_float(metrics.get("pearson"))
    if pearson is None:
        return None
    mae = float(metrics.get("mae", 6.0))
    value = (
        float(metrics.get("accuracy", 0.0))
        + float(metrics.get("macro_f1", 0.0))
        + (1.0 - mae / 6.0)
        + (pearson + 1.0) / 2.0
    ) / 4.0
    return float(value)


def confusion_rows(matrix: Iterable[Iterable[int]], scenario: str = "complete") -> list[dict[str, Any]]:
    values = np.asarray(list(matrix), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for true_index, true_name in enumerate(CLASS_NAMES):
        for pred_index, pred_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "scenario": scenario,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(values[true_index, pred_index]),
                }
            )
    return rows


def aggregate_scores(metric_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(metric_rows)
    score_values = [composite_score(row) for row in rows]
    score_values = [value for value in score_values if value is not None]
    result: dict[str, Any] = {
        "scenario_count": len(rows),
        "defined_score_count": len(score_values),
        "mean_composite_score": float(np.mean(score_values)) if score_values else None,
    }
    if rows:
        for key in ("accuracy", "macro_f1", "weighted_f1", "mae"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            result[f"mean_{key}"] = float(np.mean(values)) if values else None
    return result

