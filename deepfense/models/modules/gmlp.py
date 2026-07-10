"""gMLP building blocks (Pay Attention to MLPs).

Adapted from wedefense/wedefense/models/gmlp.py (MIT License).
"""

from typing import Optional

import torch
from torch import nn


class SpacialGatingUnit(nn.Module):
    def __init__(self, d_z: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm([d_z // 2])
        self.weight = nn.Parameter(
            torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len = z.shape[0]
        max_seq = self.weight.shape[0]
        if seq_len > max_seq:
            raise ValueError(
                f"gMLP received {seq_len} frames but backend seq_len={max_seq}. "
                f"Increase model.backend.args.seq_len in your YAML "
                f"(PartialSpoof full clips need ≥1100; reference configs use 2001)."
            )
        z1, z2 = torch.chunk(z, 2, dim=-1)

        if mask is not None:
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            assert mask.shape[2] == 1
            mask = mask[:, :, 0]

        z2 = self.norm(z2)
        weight = self.weight[:seq_len, :seq_len]
        if mask is not None:
            weight = weight * mask

        z2 = torch.einsum("ij,jbd->ibd", weight, z2) + self.bias[:seq_len, None, None]
        return z1 * z2


class GMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm([d_model])
        self.activation = nn.GELU()
        self.proj1 = nn.Linear(d_model, d_ffn)
        self.sgu = SpacialGatingUnit(d_ffn, seq_len)
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        self.size = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        shortcut = x
        x = self.norm(x)
        z = self.activation(self.proj1(x))
        z = self.sgu(z, mask)
        z = self.proj2(z)
        return z + shortcut
