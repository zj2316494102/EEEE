from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from . import INPUT_DIMS, MAX_LENGTH, MODALITIES


def _longest_missing_run(mask: Tensor) -> Tensor:
    """Return the longest false run for each row, normalized by sequence length."""

    missing = ~mask.bool()
    batch, length = missing.shape
    current = torch.zeros(batch, device=mask.device, dtype=torch.float32)
    longest = torch.zeros_like(current)
    for index in range(length):
        current = torch.where(missing[:, index], current + 1.0, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest / max(length, 1)


class RobustGatedTemporalFusion(nn.Module):
    """Mask-aware dynamic-gate fusion followed by temporal self-attention.

    The network accepts three separate feature tensors because their input
    dimensions are different.  ``masks`` has one boolean ``[B, T]`` tensor per
    modality.  A position with no available modality is excluded from the
    transformer and the attention pool, with a safe dummy position used only
    for the all-missing fallback path.
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int] | list[int] = (768, 74, 35),
        max_length: int = MAX_LENGTH,
        projection_dim: int = 128,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.15,
        use_mask_input: bool = True,
        fusion: str = "dynamic",
        use_transformer: bool = True,
        use_coverage_features: bool = True,
    ) -> None:
        super().__init__()
        if len(input_dims) != 3:
            raise ValueError("input_dims must contain text, audio and vision dimensions")
        if projection_dim % attention_heads != 0:
            raise ValueError("projection_dim must be divisible by attention_heads")
        if fusion not in {"dynamic", "mean"}:
            raise ValueError("fusion must be 'dynamic' or 'mean'")
        self.input_dims = tuple(int(value) for value in input_dims)
        self.max_length = int(max_length)
        self.projection_dim = int(projection_dim)
        self.transformer_layers = int(transformer_layers)
        self.attention_heads = int(attention_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)
        self.use_mask_input = bool(use_mask_input)
        self.fusion = fusion
        self.use_transformer = bool(use_transformer)
        self.use_coverage_features = bool(use_coverage_features)

        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, projection_dim),
                    nn.LayerNorm(projection_dim),
                    nn.GELU(),
                )
                for modality, dim in zip(MODALITIES, self.input_dims)
            }
        )
        self.gate_scores = nn.ModuleDict(
            {modality: nn.Linear(projection_dim, 1) for modality in MODALITIES}
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_length, projection_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(len(MODALITIES), projection_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

        state_dim = projection_dim // 4 if use_mask_input else 0
        self.state_embedding = nn.Embedding(8, max(state_dim, 1))
        self.fusion_projection = nn.Sequential(
            nn.Linear(projection_dim + state_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.temporal_encoder: nn.Module = nn.TransformerEncoder(
                layer, num_layers=transformer_layers
            )
        else:
            self.temporal_encoder = nn.Identity()
        self.pool_score = nn.Linear(projection_dim, 1)
        self.fallback_state = nn.Parameter(torch.zeros(projection_dim))
        head_input_dim = projection_dim + (6 if use_coverage_features else 0)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 3),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "input_dims": list(self.input_dims),
            "max_length": self.max_length,
            "projection_dim": self.projection_dim,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "use_mask_input": self.use_mask_input,
            "fusion": self.fusion,
            "use_transformer": self.use_transformer,
            "use_coverage_features": self.use_coverage_features,
        }

    def _coerce_masks(self, masks: Mapping[str, Tensor] | Tensor, device: torch.device) -> Tensor:
        if isinstance(masks, Mapping):
            result = torch.stack(
                [torch.as_tensor(masks[modality], device=device, dtype=torch.bool) for modality in MODALITIES],
                dim=1,
            )
        else:
            result = torch.as_tensor(masks, device=device, dtype=torch.bool)
        if result.ndim != 3 or result.shape[1] != 3 or result.shape[2] != self.max_length:
            raise ValueError(f"masks must have shape [B, 3, {self.max_length}], got {tuple(result.shape)}")
        return result

    def _coerce_features(self, features: Mapping[str, Tensor], device: torch.device) -> list[Tensor]:
        tensors: list[Tensor] = []
        for modality, input_dim in zip(MODALITIES, self.input_dims):
            if modality not in features:
                raise ValueError(f"Missing feature tensor: {modality}")
            tensor = torch.as_tensor(features[modality], device=device, dtype=torch.float32)
            if tensor.ndim != 3 or tensor.shape[1:] != (self.max_length, input_dim):
                raise ValueError(
                    f"{modality} must have shape [B, {self.max_length}, {input_dim}], got {tuple(tensor.shape)}"
                )
            tensors.append(tensor)
        return tensors

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | Tensor,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        mask_tensor = self._coerce_masks(masks, device)
        feature_tensors = self._coerce_features(features, device)
        network_masks = mask_tensor if self.use_mask_input else torch.ones_like(mask_tensor)
        hidden: list[Tensor] = []
        scores: list[Tensor] = []
        for index, (modality, values) in enumerate(zip(MODALITIES, feature_tensors)):
            available = network_masks[:, index].unsqueeze(-1)
            values = values.masked_fill(~available, 0.0)
            encoded = self.projections[modality](values)
            encoded = encoded + self.position_embedding[:, : self.max_length]
            encoded = encoded + self.modality_embedding[index].view(1, 1, -1)
            hidden.append(encoded)
            scores.append(self.gate_scores[modality](encoded).squeeze(-1))

        hidden_stack = torch.stack(hidden, dim=2)  # [B, T, M, D]
        available = network_masks.permute(0, 2, 1)  # [B, T, M]
        if self.fusion == "mean":
            weights = available.to(hidden_stack.dtype)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        else:
            score_stack = torch.stack(scores, dim=2)
            masked_scores = score_stack.masked_fill(~available, -1e4)
            shifted = masked_scores - masked_scores.max(dim=2, keepdim=True).values
            weights = torch.exp(shifted) * available.to(shifted.dtype)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        fused = (hidden_stack * weights.unsqueeze(-1)).sum(dim=2)

        state_code = (
            network_masks[:, 0].long()
            + 2 * network_masks[:, 1].long()
            + 4 * network_masks[:, 2].long()
        )
        if self.use_mask_input:
            state = self.state_embedding(state_code)
            fused = self.fusion_projection(torch.cat([fused, state], dim=-1))
        else:
            # The same projection is retained so checkpoints have a stable shape.
            fused = self.fusion_projection(fused)

        position_available = available.any(dim=2)
        all_unavailable = ~position_available.any(dim=1)
        padding = ~position_available
        safe_padding = padding.clone()
        if all_unavailable.any():
            safe_padding[all_unavailable, 0] = False
            fused = fused.clone()
            fused[all_unavailable] = 0.0
        if self.use_transformer:
            encoded = self.temporal_encoder(fused, src_key_padding_mask=safe_padding)
        else:
            encoded = fused
        pool_logits = self.pool_score(encoded).squeeze(-1).masked_fill(safe_padding, -1e4)
        pool_weights = torch.softmax(pool_logits, dim=1)
        pooled = (encoded * pool_weights.unsqueeze(-1)).sum(dim=1)
        if all_unavailable.any():
            pooled = pooled.clone()
            pooled[all_unavailable] = self.fallback_state

        coverage = network_masks.float().mean(dim=2)
        longest_missing = torch.stack(
            [_longest_missing_run(network_masks[:, index]) for index in range(3)], dim=1
        )
        head_input = pooled
        if self.use_coverage_features:
            head_input = torch.cat([head_input, coverage, longest_missing], dim=-1)
        logits = self.classification_head(head_input)
        intensity = 3.0 * torch.tanh(self.regression_head(head_input).squeeze(-1))
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "intensity": intensity,
            "representation": pooled,
            "gate_weights": weights,
            "coverage": coverage,
            "longest_missing": longest_missing,
            "all_unavailable": all_unavailable,
            "position_available": position_available,
        }


def build_model(config: Mapping[str, object] | None = None) -> RobustGatedTemporalFusion:
    values = dict(config or {})
    return RobustGatedTemporalFusion(
        input_dims=tuple(values.get("input_dims", (768, 74, 35))),
        max_length=int(values.get("max_length", MAX_LENGTH)),
        projection_dim=int(values.get("projection_dim", 128)),
        transformer_layers=int(values.get("transformer_layers", 2)),
        attention_heads=int(values.get("attention_heads", 4)),
        feedforward_dim=int(values.get("feedforward_dim", 256)),
        dropout=float(values.get("dropout", 0.15)),
        use_mask_input=bool(values.get("use_mask_input", True)),
        fusion=str(values.get("fusion", "dynamic")),
        use_transformer=bool(values.get("use_transformer", True)),
        use_coverage_features=bool(values.get("use_coverage_features", True)),
    )
