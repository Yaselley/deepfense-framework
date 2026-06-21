"""Weighted sum over SSL encoder layer outputs."""

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


def _hidden_states_fairseq(model, inputs: torch.Tensor, padding_mask: torch.Tensor | None) -> list[torch.Tensor]:
    res = model(
        inputs,
        padding_mask=padding_mask,
        mask=False,
        features_only=True,
    )
    layer_tensors = [_to_btc(res["features"])]
    for item in res["layer_results"]:
        layer_tensors.append(_to_btc(item[0] if isinstance(item, tuple) else item))
    return layer_tensors


def _hidden_states_unil_wavlm(model, inputs: torch.Tensor, padding_mask: torch.Tensor | None) -> list[torch.Tensor]:
    out, _ = model.extract_features(
        inputs,
        padding_mask=padding_mask,
        mask=False,
        ret_layer_results=True,
    )
    _, layer_results = out
    return [_to_btc(layer) for layer in layer_results]


def _hidden_states_huggingface(model, inputs: torch.Tensor, attention_mask: torch.Tensor | None) -> list[torch.Tensor]:
    outputs = model(inputs, attention_mask=attention_mask, output_hidden_states=True)
    return [_to_btc(h) for h in outputs.hidden_states[1:]]


def select_layer_tensors(
    layer_tensors: list[torch.Tensor],
    *,
    encoder_only: bool,
    num_layers: int | None,
    layer_indices: Any,
) -> list[torch.Tensor]:
    selected = layer_tensors[1:] if encoder_only and len(layer_tensors) > 1 else layer_tensors
    if layer_indices is not None:
        idx = parse_layer_indices(layer_indices, len(selected))
        return [selected[i] for i in idx]
    if num_layers is not None:
        selected = selected[:num_layers]
    return selected


def weighted_sum_layers(layer_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Combine ``L`` tensors ``(B, T, C)`` with normalized weights ``(L,)``."""
    if not layer_tensors:
        raise ValueError("No layer tensors to combine")
    if len(layer_tensors) != weights.shape[0]:
        raise ValueError(
            f"Expected {len(layer_tensors)} weights, got shape {tuple(weights.shape)}"
        )
    stacked = torch.stack(layer_tensors, dim=0)  # (L, B, T, C)
    view = weights.view(-1, 1, 1, 1)
    return (view * stacked).sum(dim=0)


class LayerWeightCombiner(nn.Module):
    """Learnable or fixed softmax-normalized weights over encoder layers."""

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
                logits = torch.log(torch.tensor(init_weights, dtype=torch.float32).clamp_min(1e-8))
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
                return F.softmax(self.logits, dim=0)
            if self.normalize == "none":
                return self.logits
            raise ValueError(f"Unknown normalize mode: {self.normalize}")
        return self.fixed_weights

    def forward(self, layer_tensors: list[torch.Tensor]) -> torch.Tensor:
        return weighted_sum_layers(layer_tensors, self.normalized_weights())


class LayerWeightedSumFrontendMixin:
    """Shared config + forward logic for layer-sum SSL frontends."""

    frontend_type: str = ""
    default_source: str = "fairseq"

    def _init_layer_sum_config(self, config: dict[str, Any]) -> None:
        self.encoder_only = bool(config.get("encoder_only", True))
        self.num_layers = config.get("num_layers", 24)
        self.layer_indices = config.get("layer_indices")
        self.learnable_layer_weights = bool(config.get("learnable_layer_weights", True))
        self.weight_normalize = config.get("weight_normalize", "softmax")
        init_weights = config.get("layer_weights")

        if self.layer_indices is not None:
            count = len(parse_layer_indices(self.layer_indices, 512))
        else:
            count = int(self.num_layers)

        self.layer_combiner = LayerWeightCombiner(
            count,
            learnable=self.learnable_layer_weights,
            init_weights=init_weights,
            normalize=self.weight_normalize,
        )
        logger.info(
            "Layer-weighted frontend: type=%s source=%s layers=%s learnable=%s",
            self.frontend_type,
            self.source,
            self.layer_indices if self.layer_indices is not None else f"0:{self.num_layers}",
            self.learnable_layer_weights,
        )

    def _extract_layer_tensors(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if self.source == "fairseq":
            padding_mask = mask.eq(0) if mask is not None else None
            return _hidden_states_fairseq(self.model, input_data, padding_mask)

        if self.frontend_type == "wavlm" and self.source == "unil":
            padding_mask = mask.eq(0) if mask is not None else None
            return _hidden_states_unil_wavlm(self.model, input_data, padding_mask)

        if self.source == "huggingface":
            attention_mask = mask.long() if mask is not None else None
            return _hidden_states_huggingface(self.model, input_data, attention_mask)

        raise ValueError(
            f"Unsupported layer-sum frontend: type={self.frontend_type!r}, source={self.source!r}"
        )

    def _forward_layer_sum(self, input_data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        layer_tensors = self._extract_layer_tensors(input_data, mask)
        selected = select_layer_tensors(
            layer_tensors,
            encoder_only=self.encoder_only,
            num_layers=self.num_layers,
            layer_indices=self.layer_indices,
        )
        if len(selected) != self.layer_combiner.num_layers:
            raise RuntimeError(
                f"Selected {len(selected)} layers but combiner expects "
                f"{self.layer_combiner.num_layers}. Check num_layers / layer_indices."
            )
        return self.layer_combiner(selected)
