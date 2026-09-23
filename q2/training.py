from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_torch_checkpoint,
    make_training_checkpoint,
    restore_rng_state,
)
from .data import SplitData
from .evaluation import evaluate_scenarios, evaluate_split
from .losses import distillation_loss, supervised_loss
from .masking import (
    MaskScenario,
    apply_whole_modality_dropout,
    generate_contiguous_block_masks,
    generate_random_point_masks,
    stack_masks,
    unstack_masks,
)
from .model import RobustGatedTemporalFusion, build_model
from .torchdata import make_loader


@dataclass
class TrainingResult:
    model: torch.nn.Module
    history: list[dict[str, Any]]
    best_epoch: int
    best_score: float
    seed: int
    resumed: bool = False
    resume_epoch: int = 0
    checkpoint_path: str | None = None


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def class_weights_from_training(
    split: SplitData,
    enabled: bool = True,
    mode: str = "sqrt_inverse",
) -> np.ndarray:
    mode = str(mode).lower()
    if not enabled or mode in {"none", "uniform"}:
        return np.ones(3, dtype=np.float32)
    counts = np.bincount(split.classification, minlength=3).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    if mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(safe)
    elif mode == "inverse":
        weights = 1.0 / safe
    elif mode == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - np.power(beta, safe))
    else:
        raise ValueError(
            "class_weight_mode must be one of: none, sqrt_inverse, inverse, effective"
        )
    weights /= weights.mean()
    return weights.astype(np.float32)


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _mask_batch_features(
    batch_features: Mapping[str, Tensor],
    masks: Tensor,
) -> dict[str, Tensor]:
    return {
        modality: batch_features[modality].masked_fill(
            ~masks[:, index].unsqueeze(-1), 0.0
        )
        for index, modality in enumerate(("text", "audio", "vision"))
    }


def _validation_rows(
    model: torch.nn.Module,
    valid: SplitData,
    selection_scenarios: Sequence[MaskScenario] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    if selection_scenarios:
        rows, _ = evaluate_scenarios(
            model, valid, selection_scenarios, device=device, batch_size=batch_size
        )
    else:
        complete, _ = evaluate_split(model, valid, device=device, batch_size=batch_size)
        complete["scenario"] = "complete"
        rows = [complete]
    scores = [row.get("composite_score") for row in rows if row.get("composite_score") is not None]
    if scores:
        score = float(np.mean(scores))
    else:
        # Keep Pearson undefined in persisted metrics, while still making early
        # stopping deterministic for a temporarily constant regressor.
        score = float(
            np.mean(
                [
                    (
                        float(row.get("accuracy", 0.0))
                        + float(row.get("macro_f1", 0.0))
                        + 1.0
                        - float(row.get("mae", 6.0)) / 6.0
                    )
                    / 3.0
                    for row in rows
                ]
            )
        )
    return rows, score


def train_one_model(
    train: SplitData,
    valid: SplitData,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    seed: int,
    device: str | torch.device = "cpu",
    class_weights: np.ndarray | None = None,
    use_augmentation: bool = True,
    augmentation_mode: str = "contiguous",
    teacher: RobustGatedTemporalFusion | None = None,
    selection_scenarios: Sequence[MaskScenario] | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    set_global_seed(seed)
    device_obj = torch.device(device)
    model = (model_factory() if model_factory is not None else build_model(model_config)).to(device_obj)
    if teacher is not None:
        teacher = teacher.to(device_obj).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    batch_size = int(training_config.get("batch_size", 64))
    max_epochs = int(training_config.get("max_epochs", 100))
    patience = int(training_config.get("early_stopping_patience", 12))
    learning_rate = float(training_config.get("learning_rate", 3e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    lambda_regression = float(training_config.get("lambda_regression", 1.0))
    lambda_consistency = float(training_config.get("lambda_consistency", 0.05))
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    focal_gamma = float(training_config.get("focal_gamma", 0.0))
    distill_weight = float(training_config.get("distill_weight", 0.0)) if teacher is not None else 0.0
    distill_temperature = float(training_config.get("distill_temperature", 2.0))
    distill_regression_weight = float(training_config.get("distill_regression_weight", 0.5))
    augmentation_probability = float(training_config.get("mask_augmentation_probability", 0.75))
    max_fraction = float(training_config.get("mask_max_fraction", 0.60))
    random_point_fraction = float(training_config.get("random_point_fraction", 0.20))
    whole_modality_dropout_probability = float(
        training_config.get("whole_modality_dropout_probability", 0.0)
    )
    gradient_clip = float(training_config.get("gradient_clip_norm", 1.0))
    num_workers = int(training_config.get("num_workers", 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = torch.as_tensor(
        class_weights if class_weights is not None else np.ones(3, dtype=np.float32),
        dtype=torch.float32,
        device=device_obj,
    )
    loader = make_loader(train, batch_size, shuffle=True, num_workers=num_workers, require_labels=True)
    rng = np.random.default_rng(int(seed) + 7919)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_epoch = 0
    resumed = False
    resume_epoch = 0
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    stage = str((checkpoint_metadata or {}).get("stage", checkpoint_file.stem if checkpoint_file else "student"))
    runtime_model_config = (
        model.config_dict() if hasattr(model, "config_dict") else dict(model_config)
    )

    def save_checkpoint(epoch: int, completed: bool) -> None:
        if checkpoint_file is None:
            return
        payload = make_training_checkpoint(
            stage=stage,
            seed=seed,
            epoch=epoch,
            completed=completed,
            model_config=runtime_model_config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_state_dict=best_state,
            best_epoch=best_epoch,
            best_score=best_score,
            stale_epochs=stale_epochs,
            history=history,
            rng_state=capture_rng_state(rng),
            metadata=checkpoint_metadata,
        )
        atomic_torch_save(payload, checkpoint_file)

    if resume_path is not None and Path(resume_path).exists():
        payload = load_torch_checkpoint(resume_path, device_obj)
        if int(payload.get("checkpoint_version", 0)) != 1:
            raise RuntimeError(f"Unsupported checkpoint version: {resume_path}")
        if str(payload.get("stage")) != stage:
            raise RuntimeError(
                f"Checkpoint stage mismatch: expected {stage}, got {payload.get('stage')}"
            )
        if int(payload.get("seed", seed)) != int(seed):
            raise RuntimeError(
                f"Checkpoint seed mismatch: expected {seed}, got {payload.get('seed')}"
            )
        expected_metadata = dict(checkpoint_metadata or {})
        saved_metadata = dict(payload.get("metadata") or {})
        for key in ("config_hash", "feature_version", "sample_limit"):
            if key in expected_metadata and key in saved_metadata and expected_metadata[key] != saved_metadata[key]:
                raise RuntimeError(
                    f"Checkpoint metadata mismatch for {key}: "
                    f"expected {expected_metadata[key]!r}, got {saved_metadata[key]!r}"
                )
        model.load_state_dict(payload["model_state_dict"])
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        history = list(payload.get("history") or [])
        best_epoch = int(payload.get("best_epoch", 0))
        best_score = float(payload.get("best_score", "-inf"))
        stale_epochs = int(payload.get("stale_epochs", 0))
        best_state_payload = payload.get("best_state_dict")
        best_state = None if best_state_payload is None else dict(best_state_payload)
        resume_epoch = int(payload.get("epoch", 0))
        start_epoch = resume_epoch + 1
        last_epoch = resume_epoch
        resumed = True
        restore_rng_state(payload.get("rng_state") or {}, rng)
        if bool(payload.get("completed", False)):
            if best_state is not None:
                model.load_state_dict(best_state)
            return TrainingResult(
                model=model,
                history=history,
                best_epoch=best_epoch,
                best_score=float(best_score),
                seed=int(seed),
                resumed=True,
                resume_epoch=resume_epoch,
                checkpoint_path=str(checkpoint_file) if checkpoint_file else None,
            )

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        loss_sums = {"cross_entropy": 0.0, "huber": 0.0, "consistency": 0.0, "distillation": 0.0}
        for batch in loader:
            features = _move_nested(batch["features"], device_obj)
            masks_dict = _move_nested(batch["masks"], device_obj)
            base_masks = torch.stack([masks_dict[modality] for modality in ("text", "audio", "vision")], dim=1)
            full_features = {modality: values.clone() for modality, values in features.items()}
            if use_augmentation:
                base_numpy = base_masks.detach().cpu().numpy().astype(bool)
                if augmentation_mode == "random_point":
                    augmented_numpy = generate_random_point_masks(
                        base_numpy,
                        rng,
                        probability=augmentation_probability,
                        missing_fraction=random_point_fraction,
                    )
                else:
                    augmented_numpy, _ = generate_contiguous_block_masks(
                        base_numpy,
                        rng,
                        probability=augmentation_probability,
                        max_fraction=max_fraction,
                    )
                if whole_modality_dropout_probability > 0.0:
                    augmented_numpy = apply_whole_modality_dropout(
                        augmented_numpy,
                        rng,
                        probability=whole_modality_dropout_probability,
                        keep_at_least_one_modality=True,
                    )
                augmented_masks = torch.from_numpy(augmented_numpy).to(device_obj)
            else:
                augmented_masks = base_masks
            masked_features = _mask_batch_features(features, augmented_masks)
            augmented_masks_dict = {
                modality: augmented_masks[:, index]
                for index, modality in enumerate(("text", "audio", "vision"))
            }
            output = model(masked_features, augmented_masks_dict)
            classification = batch["classification"].to(device_obj, non_blocking=True)
            regression = batch["regression"].to(device_obj, non_blocking=True)
            loss, loss_items = supervised_loss(
                output["logits"],
                output["intensity"],
                classification,
                regression,
                class_weights=weights,
                lambda_regression=lambda_regression,
                lambda_consistency=lambda_consistency,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )
            if teacher is not None and distill_weight > 0:
                with torch.no_grad():
                    teacher_output = teacher(full_features, masks_dict)
                kd, kd_items = distillation_loss(
                    teacher_output["logits"],
                    teacher_output["intensity"],
                    output["logits"],
                    output["intensity"],
                    temperature=distill_temperature,
                    regression_weight=distill_regression_weight,
                )
                loss = loss + distill_weight * kd
                loss_items["distillation"] = float(kd_items["total"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            count = int(classification.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            seen += count
            for key in ("cross_entropy", "huber", "consistency"):
                loss_sums[key] += loss_items[key] * count
            loss_sums["distillation"] += loss_items.get("distillation", 0.0) * count

        validation_rows, score = _validation_rows(
            model, valid, selection_scenarios, device_obj, batch_size
        )
        complete = next((row for row in validation_rows if row.get("scenario") == "complete"), validation_rows[0])
        record = {
            "seed": int(seed),
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "train_cross_entropy": loss_sums["cross_entropy"] / max(seen, 1),
            "train_huber": loss_sums["huber"] / max(seen, 1),
            "train_consistency": loss_sums["consistency"] / max(seen, 1),
            "train_distillation": loss_sums["distillation"] / max(seen, 1),
            "validation_selection_score": score,
            "validation_accuracy": complete.get("accuracy"),
            "validation_macro_f1": complete.get("macro_f1"),
            "validation_mae": complete.get("mae"),
            "validation_pearson": complete.get("pearson"),
        }
        history.append(record)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        # The checkpoint is written only after the complete epoch, including
        # validation, has finished. A killed process therefore resumes from a
        # known-good epoch rather than a half-written batch.
        save_checkpoint(epoch, completed=False)
        last_epoch = epoch
        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    completed_epoch = max(last_epoch, resume_epoch, best_epoch)
    save_checkpoint(completed_epoch, completed=True)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_score=float(best_score),
        seed=int(seed),
        resumed=resumed,
        resume_epoch=resume_epoch,
        checkpoint_path=str(checkpoint_file) if checkpoint_file else None,
    )
