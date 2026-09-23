from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def consistency_loss(probabilities: Tensor, intensity: Tensor) -> Tensor:
    direction = probabilities[:, 2] - probabilities[:, 0]
    target = intensity / 3.0
    return F.mse_loss(direction, target)


def representation_consistency_loss(
    native_representation: Tensor,
    masked_representation: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    """Align complete/native and additionally masked sample representations."""

    if sample_mask is not None:
        sample_mask = sample_mask.to(device=native_representation.device, dtype=torch.bool).reshape(-1)
        if not bool(sample_mask.any()):
            return native_representation.sum() * 0.0
        native_representation = native_representation[sample_mask]
        masked_representation = masked_representation[sample_mask]
    return F.mse_loss(masked_representation, native_representation.detach())


def masked_reconstruction_loss(
    reconstruction: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    synthetic_missing_masks: Tensor,
) -> Tensor:
    """Measure reconstruction only where training augmentation deleted data."""

    total: Tensor | None = None
    count = 0
    for index, modality in enumerate(("text", "audio", "vision")):
        if modality not in reconstruction or modality not in targets:
            continue
        prediction = reconstruction[modality]
        target = targets[modality].to(device=prediction.device, dtype=prediction.dtype)
        missing = synthetic_missing_masks[:, index].to(device=prediction.device, dtype=torch.bool)
        missing = missing.unsqueeze(-1).expand_as(target)
        if not bool(missing.any()):
            continue
        squared_error = F.mse_loss(prediction, target, reduction="none")
        contribution = squared_error.masked_select(missing).sum()
        total = contribution if total is None else total + contribution
        count += int(missing.sum().detach().cpu())
    if total is None or count == 0:
        if reconstruction:
            return next(iter(reconstruction.values())).sum() * 0.0
        return synthetic_missing_masks.float().sum() * 0.0
    return total / float(count)


def modality_auxiliary_loss(
    modality_logits: Tensor,
    classification: Tensor,
    modality_available: Tensor | None = None,
) -> Tensor:
    """Apply independent classification supervision to available modalities."""

    if modality_logits.ndim != 3 or modality_logits.shape[2] != 3:
        raise ValueError("modality_logits must have shape [B, 3, 3]")
    batch_size, modality_count, _ = modality_logits.shape
    targets = classification.long().view(batch_size, 1).expand(-1, modality_count)
    per_modality = F.cross_entropy(
        modality_logits.reshape(-1, 3), targets.reshape(-1), reduction="none"
    ).reshape(batch_size, modality_count)
    if modality_available is None:
        return per_modality.mean()
    available = modality_available.to(device=per_modality.device, dtype=per_modality.dtype)
    return (per_modality * available).sum() / available.sum().clamp_min(1.0)


def supervised_loss(
    logits: Tensor,
    intensity: Tensor,
    classification: Tensor,
    regression: Tensor,
    class_weights: Tensor | None = None,
    lambda_regression: float = 1.0,
    lambda_consistency: float = 0.05,
    neutral_aux_logit: Tensor | None = None,
    lambda_neutral_aux: float = 0.0,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    probabilities = torch.softmax(logits, dim=-1)
    label_smoothing = float(max(0.0, min(0.99, label_smoothing)))
    focal_gamma = float(max(0.0, focal_gamma))
    if focal_gamma > 0.0:
        per_sample = F.cross_entropy(
            logits,
            classification.long(),
            reduction="none",
            label_smoothing=label_smoothing,
        )
        focal_factor = (1.0 - torch.exp(-per_sample)).pow(focal_gamma)
        if class_weights is not None:
            sample_weights = class_weights[classification.long()]
            ce = (per_sample * focal_factor * sample_weights).mean()
        else:
            ce = (per_sample * focal_factor).mean()
    else:
        ce = F.cross_entropy(
            logits,
            classification.long(),
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    huber = F.huber_loss(intensity, regression.float(), delta=1.0)
    consistency = consistency_loss(probabilities, intensity)
    if neutral_aux_logit is not None and float(lambda_neutral_aux) > 0.0:
        neutral_target = (classification.long() == 1).to(neutral_aux_logit.dtype)
        neutral_aux = F.binary_cross_entropy_with_logits(neutral_aux_logit, neutral_target)
    else:
        neutral_aux = torch.zeros((), device=logits.device, dtype=logits.dtype)
    total = (
        ce
        + float(lambda_regression) * huber
        + float(lambda_consistency) * consistency
        + float(lambda_neutral_aux) * neutral_aux
    )
    return total, {
        "total": float(total.detach().cpu()),
        "cross_entropy": float(ce.detach().cpu()),
        "huber": float(huber.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "neutral_aux": float(neutral_aux.detach().cpu()),
    }


def distillation_loss(
    teacher_logits: Tensor,
    teacher_intensity: Tensor,
    student_logits: Tensor,
    student_intensity: Tensor,
    temperature: float = 2.0,
    regression_weight: float = 0.0,
    confidence_gated: bool = True,
    confidence_threshold: float = 0.70,
    confidence_scale: float = 0.30,
) -> tuple[Tensor, dict[str, float]]:
    temperature = max(float(temperature), 1e-3)
    teacher_probabilities = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits / temperature, dim=-1)
    per_sample_kl = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1) * temperature**2
    confidence = torch.softmax(teacher_logits.detach(), dim=-1).amax(dim=-1)
    if confidence_gated:
        scale = max(float(confidence_scale), 1e-6)
        gates = ((confidence - float(confidence_threshold)) / scale).clamp(0.0, 1.0)
    else:
        gates = torch.ones_like(confidence)
    normalizer = gates.sum().clamp_min(1e-6)
    kl = (per_sample_kl * gates).sum() / normalizer
    per_sample_regression = F.huber_loss(
        student_intensity,
        teacher_intensity.detach(),
        delta=1.0,
        reduction="none",
    )
    regression = (per_sample_regression * gates).sum() / normalizer
    total = kl + float(regression_weight) * regression
    return total, {
        "total": float(total.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "teacher_huber": float(regression.detach().cpu()),
        "teacher_confidence": float(confidence.detach().mean().cpu()),
        "distillation_gate": float(gates.detach().mean().cpu()),
    }
