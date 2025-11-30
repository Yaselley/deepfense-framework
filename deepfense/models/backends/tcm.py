# adapted from https://github.com/ductuantruong/tcm_add/blob/main/model.py

import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones

from deepfense.models.modules.conformer.conformer import ConformerBlock
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor(
        [
            [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
            for p in range(n_channels)
        ]
    )
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


@register_backend("TCM")
class TCM_Conformer(BaseBackend):
    def __init__(self, config):
        super().__init__(config)

        # Extract configuration with defaults (based on your snippets)
        self.emb_size = config.get("emb_size", 128)
        self.heads = config.get("heads", 4)
        self.ffmult = config.get("ffmult", 4)
        self.exp_fac = config.get("exp_fac", 2)
        self.kernel_size = config.get("kernel_size", 16)
        self.n_encoders = config.get("num_encoders", 1)  # Note: 'num_encoders' vs 'n_encoders'

        # Preprocessing Layers (Previously in Model class)
        # Projects 1024 (SSL) -> emb_size
        self.LL = nn.Linear(self.input_dim, self.emb_size)

        # BN2d requires (Batch, Channel, Height, Width).
        # We will treat the sequence as a 2D image with 1 channel.
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Conformer Setup
        self.dim_head = int(self.emb_size / self.heads)

        # Positional Embedding
        self.positional_emb = nn.Parameter(
            sinusoidal_embedding(10000, self.emb_size), requires_grad=False
        )

        # Encoder Blocks
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=self.emb_size,
                dim_head=self.dim_head,
                heads=self.heads,
                ff_mult=self.ffmult,
                conv_expansion_factor=self.exp_fac,
                conv_kernel_size=self.kernel_size,
            ),
            self.n_encoders,
        )

        # Class Token
        self.class_token = nn.Parameter(torch.rand(1, self.emb_size))

    def forward(self, x, **kwargs):
        """
        Args:
            x: Input tensor from SSL model [Batch_Size, Time, 1024]
        Returns:
            embedding: [Batch_Size, emb_size]
        """

        # x shape: [bs, time, 1024]
        x = self.LL(x)  # [bs, time, emb_size]

        # Prepare for BatchNorm2d: [bs, 1, time, emb_size]
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Back to [bs, time, emb_size]

        # Add PE sliced to current sequence length
        if x.size(1) > self.positional_emb.size(1):
            # Handle edge case where input is longer than max positional embedding
            x = x[:, : self.positional_emb.size(1), :]

        x = x + self.positional_emb[:, : x.size(1), :]

        # Optimized: Use expand and cat instead of loop/stack
        b, _, _ = x.shape
        cls_tokens = self.class_token.expand(b, -1, -1)  # [bs, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [bs, 1+time, emb_size]

        for layer in self.encoder_blocks:
            x, _ = layer(x)

        # Take the first token (the class token position)
        embedding = x[:, 0, :]  # [bs, emb_size]

        return embedding
