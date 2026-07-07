import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.classical_features import (
    extract_classical_features,
    resolve_spectral_dim,
    SUPPORTED_SPECTRAL_TYPES,
)
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
from deepfense.models.frontends.layer_mixing import (
    _hidden_states_fairseq,
    _hidden_states_huggingface,
    select_layers_by_index,
    weighted_sum_layers,
)
from deepfense.models.frontends.ssl_sf_fusion import SSLSFFusionHead
from deepfense.utils.registry import register_frontend

logger = logging.getLogger(__name__)


@register_frontend("wav2vec2_ssl_sf")
class Wav2Vec2SSLSFFrontend(BaseFrontend):
    """
    Wav2Vec2/XLSR frontend fused with classical cepstral features (CQCC/MFCC/LFCC).

    Implements the Multi_View / paper fusion cases:
      concat, cross_attention, semi_cross_attention, expert, gated, weighted,
      ssl_only, sf_only.
    """

    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "fairseq")
        self.fusion_mode = config.get("fusion_mode", "semi_cross_attention")
        self.ckpt_path = config.get("ckpt_path")
        if self.fusion_mode != "sf_only" and self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided unless fusion_mode is sf_only")

        self.spectral_features = config.get("spectral_features", "cqcc")
        if self.spectral_features not in SUPPORTED_SPECTRAL_TYPES:
            raise ValueError(
                f"spectral_features must be one of {SUPPORTED_SPECTRAL_TYPES}, "
                f"got {self.spectral_features!r}"
            )
        self.proj_dim = int(config.get("proj_dim", 128))
        self.ssl_layer_weighting = config.get("ssl_layer_weighting", "learned")
        self.num_ssl_layers = int(config.get("num_ssl_layers", 24))
        self.sampling_rate = int(config.get("sampling_rate", 16000))

        if self.fusion_mode == "sf_only":
            self.model = None
            self.freeze_summary = {"sf_only": True}
        else:
            if self.source == "fairseq":
                if not os.path.exists(self.ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file not found: {self.ckpt_path}")
                from deepfense.models.modules.fairseq_local import load_fairseq_model

                self.model = load_fairseq_model(self.ckpt_path)
            elif self.source == "huggingface":
                from transformers import Wav2Vec2Model

                self.model = Wav2Vec2Model.from_pretrained(self.ckpt_path)
            else:
                raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

            num_encoder_layers = len(self.model.encoder.layers)
            if self.num_ssl_layers > num_encoder_layers:
                raise ValueError(
                    f"num_ssl_layers={self.num_ssl_layers} exceeds encoder depth "
                    f"({num_encoder_layers})"
                )
            self.layer_indices = list(range(self.num_ssl_layers))
            self.freeze_summary = apply_ssl_freeze_policy(self.model, self.source, config)
            if self.ssl_layer_weighting == "learned":
                self.ssl_layer_logits = nn.Parameter(torch.zeros(self.num_ssl_layers))
            else:
                self.register_buffer("ssl_layer_logits", None)

        spectral_dim = resolve_spectral_dim(self.spectral_features)
        self.spectral_align = config.get("spectral_align", "pool2")
        self.fusion = SSLSFFusionHead(
            fusion_mode=self.fusion_mode,
            proj_dim=self.proj_dim,
            spectral_dim=spectral_dim,
            ssl_attn_heads=int(config.get("ssl_attn_heads", 8)),
            sf_attn_heads=int(config.get("sf_attn_heads", 8)),
            spectral_align=self.spectral_align,
        )

        logger.info(
            "SSL+SF frontend: spectral=%s fusion=%s proj_dim=%d ssl_weighting=%s "
            "spectral_align=%s",
            self.spectral_features,
            self.fusion_mode,
            self.proj_dim,
            self.ssl_layer_weighting,
            self.spectral_align,
        )

    @property
    def output_dim(self) -> int:
        return self.proj_dim

    def _extract_ssl_layers(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if self.source == "fairseq":
            padding_mask = mask.to(input_data.device).eq(0) if mask is not None else None
            return _hidden_states_fairseq(self.model, input_data, padding_mask)
        if self.source == "huggingface":
            attention_mask = mask.to(input_data.device).long() if mask is not None else None
            return _hidden_states_huggingface(self.model, input_data, attention_mask)
        raise ValueError(f"Unsupported source: {self.source}")

    def _pool_ssl_layers(self, layer_tensors: list[torch.Tensor]) -> torch.Tensor:
        selected = select_layers_by_index(layer_tensors, self.layer_indices)
        if self.ssl_layer_weighting == "last":
            return selected[-1]
        weights = F.softmax(self.ssl_layer_logits, dim=0)
        return weighted_sum_layers(selected, weights)

    def _extract_spectral_batch(self, input_data: torch.Tensor) -> torch.Tensor:
        waveforms = input_data.detach().cpu().numpy()
        feats = []
        for waveform in waveforms:
            audio = waveform.astype(np.float32, copy=False)
            arr = extract_classical_features(
                audio,
                self.spectral_features,
                sr=self.sampling_rate,
            )
            feats.append(torch.from_numpy(arr))
        return torch.stack(feats, dim=0).to(input_data.device, dtype=input_data.dtype)

    def forward(self, input_data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.fusion_mode == "sf_only":
            sf = self._extract_spectral_batch(input_data)
            return self.fusion(ssl=None, sf=sf)

        sf = None
        if self.fusion_mode != "ssl_only":
            sf = self._extract_spectral_batch(input_data)

        layer_tensors = self._extract_ssl_layers(input_data, mask)
        ssl = self._pool_ssl_layers(layer_tensors)
        return self.fusion(ssl=ssl, sf=sf)
