"""Multi-band SSL layer pooling with cross-attention and orthogonality penalty."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from deepfense.models.frontends.layer_mixing import (
    LayerWeightCombiner,
    LayerWeightedSumFrontendMixin,
    resolve_encoder_layer_indices,
    select_layers_by_index,
)
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy

logger = logging.getLogger(__name__)


class BandCrossAttention(nn.Module):
    """Deep-queries-shallow cross-attention with residual on the query stream."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"proj_dim={dim} must be divisible by cross_attn_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, T, D) deep band features
            key_value: (B, T, D) shallow band features
        Returns:
            (B, T, D)
        """
        attn_out, _ = self.attn(query=query, key=key_value, value=key_value)
        return query + attn_out


def pairwise_band_overlap(weights: list[torch.Tensor]) -> torch.Tensor:
    """Sum of dot products for all distinct band weight pairs."""
    penalty = torch.tensor(0.0, device=weights[0].device, dtype=weights[0].dtype)
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            penalty = penalty + torch.dot(weights[i], weights[j])
    return penalty


class CustomLayerBandsFrontendMixin(LayerWeightedSumFrontendMixin):
    """Shared logic for wav2vec2_custom_layers-style frontends."""

    def _init_custom_layers_config(self, config: dict[str, Any], layer_indices: list[int]) -> None:
        self.layer_indices = layer_indices
        self.n_bands = int(config.get("n_bands", 2))
        if self.n_bands not in (1, 2, 3):
            raise ValueError(f"n_bands must be 1, 2, or 3, got {self.n_bands}")

        self.proj_dim = int(config.get("proj_dim", 128))
        self.ssl_dim = int(config.get("ssl_dim", 1024))
        self.cross_attn_heads = int(config.get("cross_attn_heads", 8))
        self.orthogonality_lambda = float(config.get("orthogonality_lambda", 0.1))
        self.learnable_layer_weights = bool(config.get("learnable_layer_weights", True))
        self.weight_normalize = config.get("weight_normalize", "softmax")
        if self.weight_normalize not in {"softmax", "none"}:
            raise ValueError(
                f"weight_normalize must be 'softmax' or 'none', got {self.weight_normalize!r}"
            )
        init_weights = config.get("layer_weights")

        self.band_combiners = nn.ModuleList(
            [
                LayerWeightCombiner(
                    len(self.layer_indices),
                    learnable=self.learnable_layer_weights,
                    init_weights=init_weights,
                    normalize=self.weight_normalize,
                )
                for _ in range(self.n_bands)
            ]
        )

        self.band_projections = nn.ModuleList(
            [nn.Linear(self.ssl_dim, self.proj_dim) for _ in range(self.n_bands)]
        )
        self.cross_attention = (
            BandCrossAttention(self.proj_dim, num_heads=self.cross_attn_heads)
            if self.n_bands >= 2
            else None
        )

        self._last_overlap_penalty: torch.Tensor | None = None

        logger.info(
            "Custom layer-bands frontend: type=%s source=%s n_bands=%d "
            "combined_layers=%s proj_dim=%d cross_attn_heads=%d orthogonality_lambda=%g",
            self.frontend_type,
            self.source,
            self.n_bands,
            self.layer_indices,
            self.proj_dim,
            self.cross_attn_heads,
            self.orthogonality_lambda,
        )

    def _pool_bands(self, layer_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [combiner(layer_tensors) for combiner in self.band_combiners]

    def _project_bands(self, bands: list[torch.Tensor]) -> list[torch.Tensor]:
        return [proj(band) for proj, band in zip(self.band_projections, bands)]

    def _compute_overlap_penalty(self) -> None:
        if self.n_bands < 2 or self.orthogonality_lambda <= 0:
            self._last_overlap_penalty = None
            return
        weights = [combiner.normalized_weights() for combiner in self.band_combiners]
        self._last_overlap_penalty = pairwise_band_overlap(weights)

    def get_band_weights(self) -> list[torch.Tensor]:
        """Normalized layer weights per band (detached), for diagnostics/plots."""
        return [c.normalized_weights().detach() for c in self.band_combiners]

    def get_auxiliary_loss(self) -> torch.Tensor | None:
        """Orthogonality penalty λ · Σ_{i<j} (a_i · a_j) from the last forward pass."""
        if self._last_overlap_penalty is None:
            return None
        return self.orthogonality_lambda * self._last_overlap_penalty

    @property
    def output_dim(self) -> int:
        return self.proj_dim

    def _forward_custom_layers(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        all_layers = self._extract_encoder_layer_tensors(input_data, mask)
        selected = select_layers_by_index(all_layers, self.layer_indices)
        if len(selected) != self.band_combiners[0].num_layers:
            raise RuntimeError(
                f"Selected {len(selected)} layers but combiners expect "
                f"{self.band_combiners[0].num_layers}"
            )

        pooled = self._pool_bands(selected)
        projected = self._project_bands(pooled)
        self._compute_overlap_penalty()

        if self.n_bands == 1:
            return projected[0]

        # Deepest band queries shallowest band (index n-1 -> index 0).
        return self.cross_attention(projected[-1], projected[0])


def init_custom_layers_frontend(wrapper, config: dict[str, Any]) -> None:
    """Resolve layer indices, apply freeze policy, and init band combiners."""
    num_encoder_layers = len(wrapper.model.encoder.layers)
    layer_indices = resolve_encoder_layer_indices(config, num_encoder_layers)

    freeze_cfg = dict(config)
    if config.get("tune_combined_layers_only") and freeze_cfg.get("unfreeze_encoder_layers") is None:
        freeze_cfg["unfreeze_encoder_layers"] = layer_indices

    wrapper.freeze_summary = apply_ssl_freeze_policy(wrapper.model, wrapper.source, freeze_cfg)
    wrapper._init_custom_layers_config(config, layer_indices)
