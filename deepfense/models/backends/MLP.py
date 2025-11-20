import torch
import torch.nn as nn
import deepfense.models.modules.pool as pooling_modules
from deepfense.models.backends.registry import register_backend


@register_backend("Pool")
class Pool(nn.Module):
    """
    A wrapper class that only performs Pooling (no projection).
    Useful if you just want to switch between TAP, ASP, MHA, etc.
    """

    def __init__(self, config):
        super(Pool, self).__init__()
        self.input_dim = getattr(config, "input_dim", 1024)

        # Initialize pooling layer via factory
        self.pool_layer = pooling_modules.get_pooling_layer(config, self.input_dim)
        self.out_dim = self.pool_layer.get_output_dim()

    def forward(self, x, **kwargs):
        """
        x: [Batch, Time, Dim] (Standard SSL output)
        """
        # The pooling modules expect [Batch, Dim, Time]
        x = x.transpose(1, 2)
        out = self.pool_layer(x)
        return out


class TransposeBatchNorm1d(nn.Module):
    """
    Wrapper for BatchNorm1d that handles input shape [Batch, Time, Dim].
    It swaps dims to [Batch, Dim, Time] for BN, then swaps back.
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # Input x: [Batch, Time, Dim]
        x = x.transpose(1, 2)  # -> [Batch, Dim, Time]
        x = self.bn(x)
        x = x.transpose(1, 2)  # -> [Batch, Time, Dim]
        return x


@register_backend("MLP")
class MLP(nn.Module):
    """
    Flexible MLP Class with selectable Normalization:
    1. Projection (Linear -> Norm -> Activation) x N layers
    2. Pooling (Mean, Stats, ASP, MHA)
    """

    def __init__(self, config):
        super(MLP, self).__init__()

        self.input_dim = getattr(config, "input_dim", 1024)
        self.projection_dims = getattr(config, "projection", [])
        self.activation_name = getattr(config, "activation", "relu").lower()

        # Options: 'batch', 'layer', or None/'none'
        self.norm_type = getattr(config, "norm_type", "batch").lower()

        self.projection_block = nn.Sequential()
        current_dim = self.input_dim

        if len(self.projection_dims) > 0:
            layers = []

            # A. Select Activation
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

            # B. Build Layers
            for target_dim in self.projection_dims:
                # 1. Linear
                layers.append(nn.Linear(current_dim, target_dim))

                # 2. Normalization
                if "batch" in self.norm_type:
                    # Use our wrapper so we don't need manual loops in forward()
                    layers.append(TransposeBatchNorm1d(target_dim))
                elif "layer" in self.norm_type:
                    # LayerNorm works natively on [Batch, Time, Dim]
                    layers.append(nn.LayerNorm(target_dim))

                # 3. Activation
                layers.append(act_layer)

                current_dim = target_dim

            self.projection_block = nn.Sequential(*layers)

        # --- 3. Build Pooling Layer ---
        self.pool_layer = pooling_modules.get_pooling_layer(config, current_dim)
        self.final_emb_size = self.pool_layer.get_output_dim()

    def forward(self, x, **kwargs):
        """
        Args:
            x: [Batch, Time, Input_Dim]
        Returns:
            emb: [Batch, Final_Dim]
        """

        if len(self.projection_dims) > 0:
            x = self.projection_block(x)  # Output: [Batch, Time, Dim]

        # The pooling modules (TAP, ASP, etc.) expect [Batch, Dim, Time]
        x = x.transpose(1, 2)

        embedding = self.pool_layer(x)  # [Batch, Final_Dim]

        return embedding
