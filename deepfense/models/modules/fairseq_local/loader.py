"""
Load a fairseq wav2vec2 / HuBERT / XLS-R checkpoint (``.pt`` file) into the
local ``Wav2Vec2Model`` **without** any ``fairseq`` dependency.

Supported checkpoint formats
-----------------------------
* New-style (``state["cfg"]``  — OmegaConf DictConfig)
* Old-style (``state["args"]`` — argparse Namespace)
"""

import logging
from typing import Union

import torch

from .models import Wav2Vec2Model

logger = logging.getLogger(__name__)


def _get(obj, key, default=None):
    """Read a value from either a dict-like (DictConfig / dict) or a
    Namespace-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_model_cfg(state: dict) -> dict:
    """
    Extract architecture hyper-parameters from a fairseq checkpoint and return
    them as a plain dict that can be passed as ``**kwargs`` to
    ``Wav2Vec2Model()``.
    """
    # Determine where the model config lives
    if "cfg" in state and state["cfg"] is not None:
        full_cfg = state["cfg"]
        # New-style: cfg.model holds the arch params
        model_cfg = _get(full_cfg, "model", full_cfg)
        task_cfg = _get(full_cfg, "task", {})
    elif "args" in state and state["args"] is not None:
        model_cfg = state["args"]
        task_cfg = state["args"]
    else:
        raise RuntimeError(
            f"Checkpoint contains neither 'cfg' nor 'args'. Keys: {list(state.keys())}"
        )

    # Read the architecture hyperparameters with safe defaults matching the
    # fairseq wav2vec2-base defaults.
    cfg = dict(
        conv_feature_layers=_get(
            model_cfg, "conv_feature_layers",
            "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        ),
        extractor_mode=str(_get(model_cfg, "extractor_mode", "default")),
        conv_bias=bool(_get(model_cfg, "conv_bias", False)),
        encoder_embed_dim=int(_get(model_cfg, "encoder_embed_dim", 768)),
        encoder_layers=int(_get(model_cfg, "encoder_layers", 12)),
        encoder_ffn_embed_dim=int(_get(model_cfg, "encoder_ffn_embed_dim", 3072)),
        encoder_attention_heads=int(_get(model_cfg, "encoder_attention_heads", 12)),
        dropout=float(_get(model_cfg, "dropout", 0.0)),
        attention_dropout=float(_get(model_cfg, "attention_dropout", 0.0)),
        activation_dropout=float(_get(model_cfg, "activation_dropout", 0.0)),
        activation_fn=str(_get(model_cfg, "activation_fn", "gelu")),
        layer_norm_first=bool(_get(model_cfg, "layer_norm_first", False)),
        dropout_input=float(_get(model_cfg, "dropout_input", 0.0)),
        dropout_features=float(_get(model_cfg, "dropout_features", 0.0)),
        feature_grad_mult=float(_get(model_cfg, "feature_grad_mult", 1.0)),
        encoder_layerdrop=float(_get(model_cfg, "encoder_layerdrop", 0.0)),
        conv_pos=int(_get(model_cfg, "conv_pos", 128)),
        conv_pos_groups=int(_get(model_cfg, "conv_pos_groups", 16)),
        conv_pos_batch_norm=bool(_get(model_cfg, "conv_pos_batch_norm", False)),
        pos_conv_depth=int(_get(model_cfg, "pos_conv_depth", 1)),
        required_seq_len_multiple=int(_get(model_cfg, "required_seq_len_multiple", 2)),
        crop_seq_to_multiple=int(_get(model_cfg, "crop_seq_to_multiple", 1)),
    )

    return cfg


def load_fairseq_model(
    ckpt_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Wav2Vec2Model:
    """
    Load a fairseq wav2vec2 / HuBERT / XLS-R checkpoint and return a
    ready-to-use ``Wav2Vec2Model`` (inference-only, no fairseq dependency).

    Parameters
    ----------
    ckpt_path : str
        Path to the ``.pt`` checkpoint file.
    device : str or torch.device
        Where to place the loaded weights.

    Returns
    -------
    Wav2Vec2Model
        The model in ``eval()`` mode with loaded weights.
    """
    logger.info(f"Loading fairseq checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    arch_cfg = _parse_model_cfg(state)

    logger.info(
        f"  Architecture: embed_dim={arch_cfg['encoder_embed_dim']}, "
        f"layers={arch_cfg['encoder_layers']}, "
        f"heads={arch_cfg['encoder_attention_heads']}, "
        f"ffn={arch_cfg['encoder_ffn_embed_dim']}"
    )

    model = Wav2Vec2Model(**arch_cfg)

    # The checkpoint may contain pre-training-only parameters (quantizer,
    # final_proj, label_embs_concat, …).  We load with strict=False so that
    # those extra keys are simply ignored.
    model_state = state["model"]
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    # Filter noise: pre-training keys we intentionally skip
    _expected_skip = {
        "quantizer", "project_q", "final_proj", "target_glu",
        "label_embs_concat", "input_quantizer", "project_inp",
    }
    unexpected_real = [
        k for k in unexpected
        if not any(k.startswith(prefix) for prefix in _expected_skip)
    ]
    if unexpected_real:
        logger.warning(f"  Unexpected keys in checkpoint: {unexpected_real}")
    if missing:
        logger.warning(f"  Missing keys (not in checkpoint): {missing}")

    model.to(device)
    model.eval()
    logger.info("  Model loaded successfully.")
    return model
