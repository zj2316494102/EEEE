from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def consistency_loss(probabilities: Tensor, intensity: Tensor) -> Tensor:
    direction = probabilities[:, 2] - probabilities[:, 0]
    target = intensity / 3.0
    return F.mse_loss(direction, target)


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
    regression_weight: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    temperature = max(float(temperature), 1e-3)
    teacher_probabilities = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probabilities, teacher_probabilities, reduction="batchmean") * temperature**2
    regression = F.huber_loss(student_intensity, teacher_intensity.detach(), delta=1.0)
    total = kl + float(regression_weight) * regression
    return total, {
        "total": float(total.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "teacher_huber": float(regression.detach().cpu()),
    }
