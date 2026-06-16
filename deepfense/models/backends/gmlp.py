"""gMLP backend for SSL features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfense.models.base_model import BaseBackend
from deepfense.models.modules.gmlp import GMLPBlock
from deepfense.utils.registry import register_backend


class SelfWeightedPooling(nn.Module):
    """Attention-based utterance pooling (SAP)."""

    def __init__(self, feature_dim: int, num_head: int = 1, mean_only: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.num_head = num_head
        self.mm_weights = nn.Parameter(torch.empty(num_head, feature_dim))
        nn.init.kaiming_uniform_(self.mm_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        weights = torch.bmm(
            inputs,
            self.mm_weights.permute(1, 0).contiguous().unsqueeze(0).repeat(batch_size, 1, 1),
        )
        attentions = F.softmax(torch.tanh(weights), dim=1)
        weighted = torch.mul(inputs, attentions.expand_as(inputs))
        return weighted.sum(1)


@register_backend("GMLP")
class GMLPBackend(BaseBackend):
    """
    gMLP stack over SSL frame features.

    Input: ``[B, T, input_dim]``
    Output:
      - ``pooling: none`` → ``[B, T, output_dim]`` (temporal / partial deepfake)
      - ``pooling: mean|sap`` → ``[B, output_dim]`` (clip-level)
    """

    def __init__(self, config):
        super().__init__(config)

        d_model = self.input_dim
        d_ffn = config.get("d_ffn", config.get("embed_dim", -2))
        if d_ffn < 0:
            d_ffn = int(d_model / abs(d_ffn))

        seq_len = int(config.get("seq_len", 512))
        gmlp_layers = int(config.get("gmlp_layers", 1))
        self.batch_first = bool(config.get("batch_first", True))

        pooling = config.get("pooling", config.get("flag_pool", "none"))
        if pooling is None:
            pooling = "none"
        self.pooling = str(pooling).lower()
        if self.pooling == "ap":
            self.pooling = "mean"

        layers = [GMLPBlock(d_model, d_ffn, seq_len) for _ in range(gmlp_layers)]
        self.layers = nn.Sequential(*layers)

        if self.pooling == "sap":
            self.pool = SelfWeightedPooling(d_model, mean_only=True)
        else:
            self.pool = None

        self.fc = nn.Linear(d_model, d_ffn, bias=False)

        self.output_dim = int(config.get("output_dim", d_ffn))
        self.final_emb_size = self.output_dim
        if self.output_dim != d_ffn:
            self.out_proj = nn.Linear(d_ffn, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x, **kwargs):
        if self.batch_first:
            x = x.permute(1, 0, 2)
            x = self.layers(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.layers(x)

        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "sap":
            x = self.pool(x)

        x = self.fc(x)
        return self.out_proj(x)
