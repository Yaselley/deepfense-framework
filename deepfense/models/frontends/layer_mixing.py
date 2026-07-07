"""Weighted sum over selected SSL encoder layer outputs."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfense.models.frontends.freeze_policy import parse_layer_indices

logger = logging.getLogger(__name__)


def _to_btc(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure activations are ``(batch, time, channels)``."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D activation tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] > tensor.shape[1]:
        return tensor.transpose(0, 1)
    return tensor


def _hidden_states_fairseq(
    model,
    inputs: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    res = model(
        inputs,
        padding_mask=padding_mask,
        mask=False,
        features_only=True,
    )
    layers: list[torch.Tensor] = []
    for item in res["layer_results"]:
        layers.append(_to_btc(item[0] if isinstance(item, tuple) else item))
    return layers


def _hidden_states_unil_wavlm(
    model,
    inputs: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    out, _ = model.extract_features(
        inputs,
        padding_mask=padding_mask,
        mask=False,
        ret_layer_results=True,
    )
    _, layer_results = out
    return [_to_btc(layer) for layer in layer_results]


def _hidden_states_huggingface(
    model,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> list[torch.Tensor]:
    outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)
    # Skip embedding / pre-encoder state; keep encoder block outputs only.
    return [_to_btc(h) for h in outputs.hidden_states[1:]]


def resolve_encoder_layer_indices(config: dict[str, Any], num_encoder_layers: int) -> list[int]:
    """
    Resolve which encoder layers (0..N-1) participate in the weighted sum.

    Priority:
      1. ``layer_indices`` if set (ranges / unique ids), e.g. ``"2-13,15-20"`` or ``[2, 5, 9]``
      2. else ``num_layers`` from layer 0, optionally skipping index 0 when ``encoder_only: false``
    """
    if "layer_indices" in config and config["layer_indices"] is not None:
        return parse_layer_indices(config["layer_indices"], num_encoder_layers)

    num_layers = int(config.get("num_layers", num_encoder_layers))
    if num_layers > num_encoder_layers:
        raise ValueError(
            f"num_layers={num_layers} exceeds available encoder layers ({num_encoder_layers})"
        )
    return list(range(num_layers))


def select_layers_by_index(
    layer_tensors: list[torch.Tensor],
    indices: list[int],
) -> list[torch.Tensor]:
    return [layer_tensors[i] for i in indices]


def weighted_sum_layers(layer_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Combine ``L`` tensors ``(B, T, C)`` with normalized weights ``(L,)``."""
    if len(layer_tensors) != weights.shape[0]:
        raise ValueError(
            f"Expected {len(layer_tensors)} weights, got shape {tuple(weights.shape)}"
        )
    ref = layer_tensors[0]
    weights = weights.to(device=ref.device, dtype=ref.dtype)
    stacked = torch.stack(layer_tensors, dim=0)
    view = weights.view(-1, 1, 1, 1)
    return (view * stacked).sum(dim=0)


class LayerWeightCombiner(nn.Module):
    """Learnable or fixed softmax-normalized weights over selected encoder layers."""

    def __init__(
        self,
        num_layers: int,
        *,
        learnable: bool = True,
        init_weights: list[float] | None = None,
        normalize: str = "softmax",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.learnable = learnable
        self.normalize = normalize

        if init_weights is not None and len(init_weights) != num_layers:
            raise ValueError(
                f"layer_weights length {len(init_weights)} != num_layers {num_layers}"
            )

        if learnable:
            if init_weights is None:
                logits = torch.zeros(num_layers)
            else:
                init = torch.tensor(init_weights, dtype=torch.float32)
                init = init / init.sum().clamp_min(1e-8)
                logits = torch.log(init.clamp_min(1e-8))
            self.logits = nn.Parameter(logits)
        else:
            if init_weights is None:
                weights = torch.full((num_layers,), 1.0 / num_layers)
            else:
                weights = torch.tensor(init_weights, dtype=torch.float32)
                weights = weights / weights.sum().clamp_min(1e-8)
            self.register_buffer("fixed_weights", weights)

    def normalized_weights(self) -> torch.Tensor:
        if self.learnable:
            if self.normalize == "softmax":
                weights = F.softmax(self.logits, dim=0)
            elif self.normalize == "none":
                weights = self.logits
                weights = weights / weights.sum().clamp_min(1e-8)
            else:
                raise ValueError(f"Unknown normalize mode: {self.normalize}")
            return weights
        return self.fixed_weights

    def forward(self, layer_tensors: list[torch.Tensor]) -> torch.Tensor:
        weights = self.normalized_weights()
        if not torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5):
            weights = weights / weights.sum().clamp_min(1e-8)
        return weighted_sum_layers(layer_tensors, weights)


class LayerWeightedSumFrontendMixin:
    """Shared config + forward logic for layer-sum SSL frontends."""

    frontend_type: str = ""
    default_source: str = "fairseq"

    def _init_layer_sum_config(self, config: dict[str, Any], layer_indices: list[int]) -> None:
        self.layer_indices = layer_indices
        self.learnable_layer_weights = bool(config.get("learnable_layer_weights", True))
        self.weight_normalize = config.get("weight_normalize", "softmax")
        if self.weight_normalize not in {"softmax", "none"}:
            raise ValueError(
                f"weight_normalize must be 'softmax' or 'none', got {self.weight_normalize!r}"
            )
        init_weights = config.get("layer_weights")

        self.layer_combiner = LayerWeightCombiner(
            len(self.layer_indices),
            learnable=self.learnable_layer_weights,
            init_weights=init_weights,
            normalize=self.weight_normalize,
        )

        logger.info(
            "Layer-sum frontend: type=%s source=%s combined_layers=%s learnable_weights=%s",
            self.frontend_type,
            self.source,
            self.layer_indices,
            self.learnable_layer_weights,
        )

    def _extract_encoder_layer_tensors(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if self.source == "fairseq":
            padding_mask = mask.to(input_data.device).eq(0) if mask is not None else None
            return _hidden_states_fairseq(self.model, input_data, padding_mask)

        if self.frontend_type == "wavlm" and self.source == "unil":
            padding_mask = mask.to(input_data.device).eq(0) if mask is not None else None
            return _hidden_states_unil_wavlm(self.model, input_data, padding_mask)

        if self.source == "huggingface":
            attention_mask = mask.to(input_data.device).long() if mask is not None else None
            return _hidden_states_huggingface(self.model, input_data, attention_mask)

        raise ValueError(
            f"Unsupported layer-sum frontend: type={self.frontend_type!r}, source={self.source!r}"
        )

    def _forward_layer_sum(self, input_data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        all_layers = self._extract_encoder_layer_tensors(input_data, mask)
        selected = select_layers_by_index(all_layers, self.layer_indices)
        if len(selected) != self.layer_combiner.num_layers:
            raise RuntimeError(
                f"Selected {len(selected)} layers but combiner expects "
                f"{self.layer_combiner.num_layers}"
            )
        return self.layer_combiner(selected)
