"""Per-frame MLP backend that preserves temporal resolution."""

import torch.nn as nn

from deepfense.models.backends.mlp import TransposeBatchNorm1d
from deepfense.models.base_model import BaseBackend
from deepfense.utils.registry import register_backend


@register_backend("FrameMLP")
class FrameMLP(BaseBackend):
    """MLP projection stack without temporal pooling. Output shape: (B, T, D)."""

    def __init__(self, config):
        super().__init__(config)
        self.projection_dims = config.get("projection", [])
        self.activation_name = config.get("activation", "relu").lower()
        self.norm_type = config.get("norm_type", "layer").lower()

        self.projection_block = nn.Sequential()
        current_dim = self.input_dim

        if len(self.projection_dims) > 0:
            layers = []
            if self.activation_name == "relu":
                act_layer = nn.ReLU(inplace=True)
            elif self.activation_name == "selu":
                act_layer = nn.SELU(inplace=True)
            elif self.activation_name == "tanh":
                act_layer = nn.Tanh()
            elif self.activation_name == "sigmoid":
                act_layer = nn.Sigmoid()
            else:
                act_layer = nn.ReLU(inplace=True)

            for target_dim in self.projection_dims:
                layers.append(nn.Linear(current_dim, target_dim))
                if "batch" in self.norm_type:
                    layers.append(TransposeBatchNorm1d(target_dim))
                elif "layer" in self.norm_type:
                    layers.append(nn.LayerNorm(target_dim))
                layers.append(act_layer)
                current_dim = target_dim

            self.projection_block = nn.Sequential(*layers)

        self.final_emb_size = current_dim
        self.output_dim = config.get("output_dim", self.final_emb_size)
        if self.output_dim != self.final_emb_size:
            self.final_proj = nn.Linear(self.final_emb_size, self.output_dim)
            self.final_emb_size = self.output_dim
        else:
            self.final_proj = nn.Identity()

    def forward(self, x, **kwargs):
        if len(self.projection_dims) > 0:
            x = self.projection_block(x)
        return self.final_proj(x)
