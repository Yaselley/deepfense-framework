"""Selective freeze / unfreeze policies for SSL frontends."""

from __future__ import annotations

import logging
import re
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical encoder sub-module tags (aligned with weight-analysis naming).
_MODULE_ALIASES: dict[str, str] = {
    "q": "attn_q",
    "k": "attn_k",
    "v": "attn_v",
    "attn_q": "attn_q",
    "attn_k": "attn_k",
    "attn_v": "attn_v",
    "attn_out": "attn_out",
    "out": "attn_out",
    "ffn_in": "ffn_in",
    "ffn_out": "ffn_out",
    "layernorm": "layernorm",
    "ln": "layernorm",
}

# Shorthand groups expanded before alias lookup.
_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "qk": ("q", "k"),
    "qkv": ("q", "k", "v"),
    "mhsa": ("q", "k", "v", "out"),
    "attn_qk": ("attn_q", "attn_k"),
    "attn_qkv": ("attn_q", "attn_k", "attn_v"),
    "attn_mhsa": ("attn_q", "attn_k", "attn_v", "attn_out"),
    "ffn": ("ffn_in", "ffn_out"),
}


def _expand_layer_token(token: str, num_layers: int) -> list[int]:
    """Expand one token: '12', '2:14', or '2-13'."""
    part = token.strip()
    if not part:
        return []
    if ":" in part:
        start_s, end_s = part.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else num_layers
        return list(range(start, end))
    if "-" in part:
        start_s, end_s = part.split("-", 1)
        return list(range(int(start_s), int(end_s) + 1))
    return [int(part)]


def parse_layer_indices(spec: Any, num_layers: int) -> list[int]:
    """Parse 0-based encoder layer indices from list, int, or string.

    Supported string forms (comma-separated tokens):
      - single index: ``"12"``
      - inclusive range: ``"2-13"``  -> layers 2..13
      - half-open slice: ``"2:14"``  -> layers 2..13
      - multiple ranges: ``"2-13,15-20"`` or ``"2:14,15:21"``
    """
    if spec is None:
        return []

    if isinstance(spec, bool):
        raise ValueError("layer spec must be a list, int, or string — not bool")

    indices: list[int] = []
    if isinstance(spec, int):
        indices = [spec]
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            if isinstance(item, str):
                indices.extend(_expand_layer_token(item, num_layers))
            else:
                indices.append(int(item))
    elif isinstance(spec, str):
        text = spec.strip()
        if not text:
            return []
        for token in text.split(","):
            indices.extend(_expand_layer_token(token, num_layers))
    else:
        raise TypeError(f"layer spec must be list/int/str, got {type(spec).__name__}")

    out = sorted(set(indices))
    for idx in out:
        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"Encoder layer index {idx} out of range [0, {num_layers - 1}] "
                f"({num_layers} encoder layers total)"
            )
    return out


def classify_encoder_param(name: str) -> str:
    """Classify a parameter name inside one encoder layer (fairseq / unil / HF)."""
    if "q_proj" in name and ("self_attn" in name or "attention" in name):
        return "attn_q"
    if "k_proj" in name and ("self_attn" in name or "attention" in name):
        return "attn_k"
    if "v_proj" in name and ("self_attn" in name or "attention" in name):
        return "attn_v"
    if "out_proj" in name and ("self_attn" in name or "attention" in name):
        return "attn_out"
    if (
        "feed_forward.intermediate_dense" in name
        or ".fc1." in name
        or name.startswith("fc1.")
    ):
        return "ffn_in"
    if (
        "feed_forward.output_dense" in name
        or ".fc2." in name
        or name.startswith("fc2.")
    ):
        return "ffn_out"
    if "final_layer_norm" in name or "layer_norm" in name or "self_attn_layer_norm" in name:
        return "layernorm"
    return "other"


def parse_module_spec(spec: Any) -> list[str]:
    """Parse encoder sub-module tags from list/int/str.

    Examples: ``"qk"``, ``"attn_q,attn_k"``, ``["attn_q", "attn_k"]``.
    """
    if spec is None:
        return []

    if isinstance(spec, bool):
        raise ValueError("module spec must be a list or string — not bool")

    tokens: list[str] = []
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return []
        tokens = [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]
    elif isinstance(spec, (list, tuple)):
        for item in spec:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in re.split(r"[,\s]+", item) if t.strip())
            else:
                raise TypeError(f"module spec items must be strings, got {type(item).__name__}")
    else:
        raise TypeError(f"module spec must be list/str, got {type(spec).__name__}")

    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in _MODULE_GROUPS:
            expanded = _MODULE_GROUPS[key]
        else:
            expanded = (key,)
        for part in expanded:
            canonical = _MODULE_ALIASES.get(part, part)
            if canonical not in seen:
                seen.add(canonical)
                out.append(canonical)

    unknown = [m for m in out if m not in set(_MODULE_ALIASES.values())]
    if unknown:
        allowed = sorted(set(_MODULE_ALIASES.values()) | set(_MODULE_GROUPS))
        raise ValueError(
            f"Unknown unfreeze_encoder_modules token(s): {unknown}. "
            f"Allowed: {', '.join(allowed)}"
        )
    return out


def _unfreeze_encoder_layer_modules(layer: nn.Module, modules: set[str]) -> None:
    for name, param in layer.named_parameters():
        if classify_encoder_param(name) in modules:
            param.requires_grad = True


def get_encoder_layers(model: nn.Module, source: str) -> nn.ModuleList:
    if source in {"fairseq", "unil"}:
        return model.encoder.layers
    if source == "huggingface":
        return model.encoder.layers
    raise ValueError(f"Unsupported frontend source for layer freeze: {source!r}")


def get_feature_extractor(model: nn.Module, source: str) -> nn.Module:
    if source in {"fairseq", "unil", "huggingface"}:
        return model.feature_extractor
    raise ValueError(f"Unsupported frontend source for layer freeze: {source!r}")


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def _unfreeze_encoder_stem(model: nn.Module, source: str) -> None:
    """Unfreeze positional conv and pre/post encoder norms (not transformer blocks)."""
    if source in {"fairseq", "unil"}:
        for name in ("layer_norm", "post_extract_proj"):
            mod = getattr(model, name, None)
            if mod is not None:
                _set_requires_grad(mod, True)
        encoder = model.encoder
        for name in ("pos_conv", "layer_norm"):
            mod = getattr(encoder, name, None)
            if mod is not None:
                _set_requires_grad(mod, True)
    elif source == "huggingface":
        for name in ("layer_norm", "feature_projection"):
            mod = getattr(model, name, None)
            if mod is not None:
                _set_requires_grad(mod, True)
        encoder = model.encoder
        for name in ("pos_conv_embed", "layer_norm"):
            mod = getattr(encoder, name, None)
            if mod is not None:
                _set_requires_grad(mod, True)


def count_trainable(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def apply_ssl_freeze_policy(model: nn.Module, source: str, config: dict) -> dict:
    """
    Configure ``requires_grad`` for an SSL frontend.

    Config keys (under ``frontend.args``):
      - ``freeze`` (bool): freeze entire SSL model when no selective unfreeze is set.
      - ``unfreeze_encoder_layers``: list/int/str of 0-based encoder layer indices to train.
        Examples: ``[22, 23]``, ``"20:24"``, ``"2-13,15-20"``.
      - ``unfreeze_encoder_modules``: optional sub-module filter inside selected layers.
        When set, only those parts of each listed layer are trained (e.g. ``"qk"`` or
        ``"attn_q,attn_k"``). Requires ``unfreeze_encoder_layers``.
      - ``unfreeze_feature_extractor`` (bool): also train the CNN frontend.
      - ``unfreeze_encoder_stem`` (bool): also train pos_conv + layer norms outside blocks.

    Priority:
      1. If selective unfreeze flags are set -> freeze all, then unfreeze those parts.
      2. Elif ``freeze: true`` -> freeze entire SSL model.
      3. Else -> train entire SSL model (legacy default).
    """
    freeze_all = bool(config.get("freeze", False))
    layer_spec = config.get("unfreeze_encoder_layers")
    module_spec = config.get("unfreeze_encoder_modules")
    unfreeze_cnn = bool(config.get("unfreeze_feature_extractor", False))
    unfreeze_stem = bool(config.get("unfreeze_encoder_stem", False))

    enc_layers = get_encoder_layers(model, source)
    num_layers = len(enc_layers)
    trainable_layer_indices = parse_layer_indices(layer_spec, num_layers) if layer_spec is not None else []
    trainable_modules = parse_module_spec(module_spec)
    if trainable_modules and not trainable_layer_indices:
        raise ValueError(
            "unfreeze_encoder_modules requires unfreeze_encoder_layers "
            "(module-only tuning must specify which layers)"
        )
    trainable_module_set = set(trainable_modules)
    has_selective = bool(trainable_layer_indices) or unfreeze_cnn or unfreeze_stem

    if not has_selective and not freeze_all:
        _set_requires_grad(model, True)
        mode = "train_all"
    else:
        _set_requires_grad(model, False)
        if has_selective:
            for idx in trainable_layer_indices:
                if trainable_module_set:
                    _unfreeze_encoder_layer_modules(enc_layers[idx], trainable_module_set)
                else:
                    _set_requires_grad(enc_layers[idx], True)
            if unfreeze_cnn:
                _set_requires_grad(get_feature_extractor(model, source), True)
            if unfreeze_stem:
                _unfreeze_encoder_stem(model, source)
            mode = "selective"
        else:
            mode = "freeze_all"

    trainable, total = count_trainable(model)
    summary = {
        "mode": mode,
        "source": source,
        "num_encoder_layers": num_layers,
        "trainable_encoder_layers": trainable_layer_indices,
        "trainable_encoder_modules": trainable_modules,
        "unfreeze_feature_extractor": unfreeze_cnn,
        "unfreeze_encoder_stem": unfreeze_stem,
        "trainable_params": trainable,
        "total_params": total,
    }
    logger.info(
        "SSL freeze policy: mode=%s source=%s trainable_layers=%s trainable_modules=%s "
        "trainable_params=%s/%s (%.2f%%)",
        mode,
        source,
        trainable_layer_indices or "none",
        trainable_modules or "all",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total else 0.0,
    )
    return summary
