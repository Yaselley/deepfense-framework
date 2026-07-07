"""Helpers shared by layer-sum frontends."""

from __future__ import annotations

from typing import Any

from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
from deepfense.models.frontends.layer_mixing import resolve_encoder_layer_indices


def init_layer_sum_frontend(wrapper, config: dict[str, Any]) -> None:
    """Resolve layer indices, apply freeze policy, and init the combiner."""
    num_encoder_layers = len(wrapper.model.encoder.layers)
    layer_indices = resolve_encoder_layer_indices(config, num_encoder_layers)

    freeze_cfg = dict(config)
    if config.get("tune_combined_layers_only") and freeze_cfg.get("unfreeze_encoder_layers") is None:
        freeze_cfg["unfreeze_encoder_layers"] = layer_indices

    wrapper.freeze_summary = apply_ssl_freeze_policy(wrapper.model, wrapper.source, freeze_cfg)
    wrapper._init_layer_sum_config(config, layer_indices)
