from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import torch
from torch import Tensor, nn

from . import INPUT_DIMS, MAX_LENGTH, MODALITIES
from .data import SplitData


def restrict_to_modality(split: SplitData, modality: str) -> SplitData:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    features = {name: split.features[name].copy() for name in MODALITIES}
    masks = {name: split.masks[name].copy() for name in MODALITIES}
    for name in MODALITIES:
        if name != modality:
            features[name].fill(0.0)
            masks[name].fill(False)
    return replace(split, name=f"{split.name}_{modality}_only", features=features, masks=masks)


class ConcatMLP(nn.Module):
    """Lightweight ordinary concatenation baseline over temporal means."""

    def __init__(
        self,
        input_dims: tuple[int, int, int] = (768, 74, 35),
        max_length: int = MAX_LENGTH,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dims = tuple(int(value) for value in input_dims)
        self.max_length = int(max_length)
        input_dim = sum(self.input_dims) + len(MODALITIES)
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classification_head = nn.Linear(hidden_dim, 3)
        self.regression_head = nn.Linear(hidden_dim, 1)

    def config_dict(self) -> dict[str, object]:
        return {
            "type": "concat_mlp",
            "input_dims": list(self.input_dims),
            "max_length": self.max_length,
            "hidden_dim": self.backbone[1].out_features,
        }

    def forward(self, features: Mapping[str, Tensor], masks: Mapping[str, Tensor] | Tensor) -> dict[str, Tensor]:
        if isinstance(masks, Mapping):
            mask = torch.stack([masks[modality].bool() for modality in MODALITIES], dim=1)
        else:
            mask = masks.bool()
        summaries = []
        for index, modality in enumerate(MODALITIES):
            values = features[modality].masked_fill(~mask[:, index].unsqueeze(-1), 0.0)
            denominator = mask[:, index].sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
            summaries.append(values.sum(dim=1) / denominator)
        coverage = mask.float().mean(dim=2)
        pooled = torch.cat([*summaries, coverage], dim=-1)
        representation = self.backbone(pooled)
        logits = self.classification_head(representation)
        intensity = 3.0 * torch.tanh(self.regression_head(representation).squeeze(-1))
        weights = mask.permute(0, 2, 1).float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        position_available = mask.any(dim=1)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "intensity": intensity,
            "representation": representation,
            "gate_weights": weights,
            "coverage": coverage,
            "longest_missing": torch.zeros_like(coverage),
            "all_unavailable": ~position_available.any(dim=1),
            "position_available": position_available,
        }

