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
        # Auxiliary Neutral/Non-neutral supervision.  The final prediction
        # remains the direct three-class head; this auxiliary head never
        # replaces the class probabilities.
        self.neutral_aux_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
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
        neutral_aux_logit = self.neutral_aux_head(head_input).squeeze(-1)
        intensity = 3.0 * torch.tanh(self.regression_head(head_input).squeeze(-1))
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "intensity": intensity,
            "neutral_aux_logit": neutral_aux_logit,
            "representation": pooled,
            "gate_weights": weights,
            "coverage": coverage,
            "longest_missing": longest_missing,
            "all_unavailable": all_unavailable,
            "position_available": position_available,
        }


class RobustLateFusionV2(nn.Module):
    """Statistics-enhanced late-fusion model for local continuous missingness.

    Each modality first produces a masked temporal-statistics representation.
    Modality experts are fused at sample level using availability-aware
    reliability weights.  The public output matches the original model so the
    existing training, checkpoint and scenario-evaluation code can be reused.
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int] | list[int] = (768, 74, 35),
        max_length: int = MAX_LENGTH,
        hidden_dim: int = 96,
        expert_dim: int = 128,
        dropout: float = 0.12,
        use_coverage_features: bool = True,
        classifier_mode: str = "two_stage",
    ) -> None:
        super().__init__()
        if len(input_dims) != 3:
            raise ValueError("input_dims must contain text, audio and vision dimensions")
        if classifier_mode not in {"direct", "two_stage"}:
            raise ValueError("classifier_mode must be 'direct' or 'two_stage'")
        self.input_dims = tuple(int(value) for value in input_dims)
        self.max_length = int(max_length)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.dropout = float(dropout)
        self.use_coverage_features = bool(use_coverage_features)
        self.classifier_mode = str(classifier_mode)

        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for modality, dim in zip(MODALITIES, self.input_dims)
            }
        )
        # mean, max, std, first, last and mean temporal difference
        statistics_dim = hidden_dim * 6
        metadata_dim = 3  # coverage, longest missing run and has-any flag
        expert_input_dim = statistics_dim + metadata_dim
        self.experts = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(expert_input_dim),
                    nn.Linear(expert_input_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, expert_dim),
                    nn.GELU(),
                )
                for modality in MODALITIES
            }
        )
        self.reliability_heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(expert_dim + metadata_dim),
                    nn.Linear(expert_dim + metadata_dim, max(expert_dim // 2, 16)),
                    nn.GELU(),
                    nn.Linear(max(expert_dim // 2, 16), 1),
                )
                for modality in MODALITIES
            }
        )
        self.expert_class_heads = nn.ModuleDict(
            {modality: nn.Linear(expert_dim, 3) for modality in MODALITIES}
        )

        joint_metadata_dim = 9  # coverage, longest missing and has-any per modality
        joint_input_dim = expert_dim * (len(MODALITIES) + 1) + 3 * len(MODALITIES) + joint_metadata_dim
        joint_dim = max(expert_dim, 128)
        self.joint = nn.Sequential(
            nn.LayerNorm(joint_input_dim),
            nn.Linear(joint_input_dim, joint_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim, joint_dim),
            nn.GELU(),
        )
        self.classification_head = nn.Linear(joint_dim, 3)
        self.neutral_head = nn.Linear(joint_dim, 1)
        self.polar_head = nn.Linear(joint_dim, 2)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, max(joint_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(joint_dim // 2, 32), 1),
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "type": "late_fusion_v2",
            "input_dims": list(self.input_dims),
            "max_length": self.max_length,
            "hidden_dim": self.hidden_dim,
            "expert_dim": self.expert_dim,
            "dropout": self.dropout,
            "use_coverage_features": self.use_coverage_features,
            "classifier_mode": self.classifier_mode,
        }

    @staticmethod
    def _coerce_masks(masks: Mapping[str, Tensor] | Tensor, device: torch.device) -> Tensor:
        if isinstance(masks, Mapping):
            result = torch.stack(
                [torch.as_tensor(masks[modality], device=device, dtype=torch.bool) for modality in MODALITIES],
                dim=1,
            )
        else:
            result = torch.as_tensor(masks, device=device, dtype=torch.bool)
        if result.ndim != 3 or result.shape[1] != 3:
            raise ValueError(f"masks must have shape [B, 3, T], got {tuple(result.shape)}")
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

    @staticmethod
    def _masked_statistics(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return statistics, coverage and longest missing run for one modality."""

        mask = mask.bool()
        available = mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        has_any = mask.any(dim=1)
        mean = (values.masked_fill(~available, 0.0).sum(dim=1) / count)
        centered = values - mean.unsqueeze(1)
        std = torch.sqrt(
            (centered.pow(2).masked_fill(~available, 0.0).sum(dim=1) / count).clamp_min(1e-8)
        )
        max_values = values.masked_fill(~available, -1e4).amax(dim=1)
        max_values = torch.where(has_any.unsqueeze(-1), max_values, torch.zeros_like(max_values))

        positions = torch.arange(values.shape[1], device=values.device).view(1, -1).expand(values.shape[0], -1)
        first_index = torch.where(mask, positions, torch.full_like(positions, values.shape[1])).amin(dim=1)
        last_index = torch.where(mask, positions, torch.full_like(positions, -1)).amax(dim=1)
        first = values[torch.arange(values.shape[0], device=values.device), first_index.clamp_max(values.shape[1] - 1)]
        last = values[torch.arange(values.shape[0], device=values.device), last_index.clamp_min(0)]
        first = torch.where(has_any.unsqueeze(-1), first, torch.zeros_like(first))
        last = torch.where(has_any.unsqueeze(-1), last, torch.zeros_like(last))

        if values.shape[1] > 1:
            pair_mask = mask[:, 1:] & mask[:, :-1]
            deltas = values[:, 1:] - values[:, :-1]
            pair_count = pair_mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
            delta_mean = deltas.masked_fill(~pair_mask.unsqueeze(-1), 0.0).sum(dim=1) / pair_count
        else:
            delta_mean = torch.zeros_like(mean)
        stats = torch.cat([mean, max_values, std, first, last, delta_mean], dim=-1)

        current = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
        longest = torch.zeros_like(current)
        missing = ~mask
        for index in range(values.shape[1]):
            current = torch.where(missing[:, index], current + 1.0, torch.zeros_like(current))
            longest = torch.maximum(longest, current)
        longest = longest / max(values.shape[1], 1)
        coverage = mask.float().mean(dim=1)
        metadata = torch.stack([coverage, longest, has_any.float()], dim=1)
        return stats, metadata, has_any

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | Tensor,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        mask_tensor = self._coerce_masks(masks, device)
        feature_tensors = self._coerce_features(features, device)
        expert_values: list[Tensor] = []
        metadata_values: list[Tensor] = []
        availability: list[Tensor] = []
        for index, modality in enumerate(MODALITIES):
            values = feature_tensors[index]
            current_mask = mask_tensor[:, index]
            values = values.masked_fill(~current_mask.unsqueeze(-1), 0.0)
            encoded = self.projections[modality](values)
            statistics, metadata, has_any = self._masked_statistics(encoded, current_mask)
            expert_input = torch.cat([statistics, metadata], dim=-1)
            expert = self.experts[modality](expert_input) * has_any.unsqueeze(-1).to(encoded.dtype)
            expert_values.append(expert)
            metadata_values.append(metadata)
            availability.append(has_any)

        experts = torch.stack(expert_values, dim=1)  # [B, M, E]
        metadata = torch.stack(metadata_values, dim=1)  # [B, M, 3]
        available = torch.stack(availability, dim=1)  # [B, M]
        reliability_scores = torch.cat(
            [
                self.reliability_heads[modality](torch.cat([expert_values[index], metadata[:, index]], dim=-1))
                for index, modality in enumerate(MODALITIES)
            ],
            dim=1,
        ).squeeze(-1)
        masked_scores = reliability_scores.masked_fill(~available, -1e4)
        reliability = torch.softmax(masked_scores, dim=1) * available.float()
        reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (experts * reliability.unsqueeze(-1)).sum(dim=1)

        expert_logits = torch.stack(
            [self.expert_class_heads[modality](expert_values[index]) for index, modality in enumerate(MODALITIES)],
            dim=1,
        )
        expert_logits = expert_logits * available.unsqueeze(-1).float()
        joint_input = torch.cat(
            [experts.flatten(start_dim=1), fused, expert_logits.flatten(start_dim=1), metadata.flatten(start_dim=1)],
            dim=-1,
        )
        representation = self.joint(joint_input)
        if self.classifier_mode == "two_stage":
            neutral_probability = torch.sigmoid(self.neutral_head(representation).squeeze(-1))
            polar_probability = torch.softmax(self.polar_head(representation), dim=-1)
            non_neutral = 1.0 - neutral_probability
            probabilities = torch.stack(
                [non_neutral * polar_probability[:, 0], neutral_probability, non_neutral * polar_probability[:, 1]],
                dim=1,
            )
            logits = torch.log(probabilities.clamp_min(1e-8))
        else:
            logits = self.classification_head(representation)
            probabilities = torch.softmax(logits, dim=-1)
        intensity = 3.0 * torch.tanh(self.regression_head(representation).squeeze(-1))

        position_available = mask_tensor.any(dim=1)
        gate_weights = reliability.unsqueeze(1).expand(-1, self.max_length, -1)
        gate_weights = gate_weights * mask_tensor.permute(0, 2, 1).float()
        gate_weights = gate_weights / gate_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        all_unavailable = ~position_available.any(dim=1)
        coverage = metadata[:, :, 0]
        longest_missing = metadata[:, :, 1]
        return {
            "logits": logits,
            "probabilities": probabilities,
            "intensity": intensity,
            "representation": representation,
            "gate_weights": gate_weights,
            "coverage": coverage,
            "longest_missing": longest_missing,
            "all_unavailable": all_unavailable,
            "position_available": position_available,
            "reliability": reliability,
        }


def build_model(config: Mapping[str, object] | None = None) -> RobustGatedTemporalFusion:
    values = dict(config or {})
    model_type = str(values.get("type", values.get("model_type", "dynamic_gate"))).lower()
    if model_type in {"late_fusion_v2", "v2", "late_fusion"}:
        return RobustLateFusionV2(
            input_dims=tuple(values.get("input_dims", (768, 74, 35))),
            max_length=int(values.get("max_length", MAX_LENGTH)),
            hidden_dim=int(values.get("hidden_dim", values.get("v2_hidden_dim", 96))),
            expert_dim=int(values.get("expert_dim", values.get("v2_expert_dim", 128))),
            dropout=float(values.get("dropout", 0.12)),
            use_coverage_features=bool(values.get("use_coverage_features", True)),
            classifier_mode=str(values.get("classifier_mode", "two_stage")),
        )
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
