import torch
import torch.nn as nn
import math
import deepfense.models.modules.pool as pooling_modules
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend

# --- Helpers (SEModule & Bottle2neck) ---


class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8
    ):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


# --- Main Backend Class ---


@register_backend("Nes2Net")
class Nes2Net(BaseBackend):
    def __init__(self, config):
        super().__init__(config)

        # 1. Config Extraction

        # Nes_ratio is a list: [Outer_Scale, Inner_Scale]
        # Default [8, 8] based on your snippet
        self.nes_ratio = config.get("nes_ratio", [8, 8])
        self.dilation = config.get("dilation", 2)
        self.se_ratio = config.get("se_ratio", 8)

        # 2. Validation
        # The input channels must be divisible by the outer scale
        if self.input_dim % self.nes_ratio[0] != 0:
            raise ValueError(
                f"Input dim {self.input_dim} must be divisible by Nes_ratio[0] {self.nes_ratio[0]}"
            )

        # 3. Build Nested Res2Net Blocks
        # We split the input dim into chunks of size C
        self.C = self.input_dim // self.nes_ratio[0]

        Build_in_Res2Nets = []
        bns = []

        # Create the parallel branches (Nes_ratio[0] - 1 branches usually processed)
        for i in range(self.nes_ratio[0] - 1):
            Build_in_Res2Nets.append(
                Bottle2neck(
                    inplanes=self.C,
                    planes=self.C,
                    kernel_size=3,
                    dilation=self.dilation,
                    scale=self.nes_ratio[1],
                    SE_ratio=self.se_ratio,
                )
            )
            bns.append(nn.BatchNorm1d(self.C))

        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)

        # Post-processing
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.relu = nn.ReLU()

        # 4. Pooling Layer
        # Uses the factory from modules.py (supports 'mean', 'stats', 'asp', etc.)
        self.pool_layer = pooling_modules.get_pooling_layer(config, self.input_dim)
        self.out_dim = self.pool_layer.get_output_dim()

    def forward(self, x, **kwargs):
        """
        Args:
            x: [Batch, Time, Dim] (Standard SSL output)
        Returns:
            embedding: [Batch, Out_Dim]
        """

        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # Split input channels
        spx = torch.split(x, self.C, dim=1)

        out = None

        # 3. Apply Nested Logic
        for i in range(self.nes_ratio[0] - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            # Apply Bottle2neck block
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)

        # Concatenate the last split (which usually skips processing in Res2Net logic)
        out = torch.cat((out, spx[-1]), dim=1)

        # Final Norm & Act
        out = self.bn(out)
        out = self.relu(out)

        # Pooling
        # Pool layer expects [B, C, T] which matches current shape `out`
        embedding = self.pool_layer(out)

        return embedding
