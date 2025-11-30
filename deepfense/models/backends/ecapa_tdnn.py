import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend
import deepfense.models.modules.pool as pooling_modules

class Res2NetBlock(nn.Module):
    """
    Res2Net Block for 1D data (Audio).
    Adapts the 2D Res2Net architecture to 1D Conv.
    """
    def __init__(self, in_planes, out_planes, scale=8, kernel_size=3, dilation=1):
        super(Res2NetBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.scale = scale
        
        # Width of each scale group
        self.width = out_planes // scale
        
        self.conv1 = nn.Conv1d(in_planes, self.width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.width * scale)
        
        self.nums = scale - 1
        convs = []
        bns = []
        
        # Construct the scale branches
        for i in range(self.nums):
            convs.append(nn.Conv1d(self.width, self.width, kernel_size=kernel_size, padding=1, dilation=dilation))
            bns.append(nn.BatchNorm1d(self.width))
            
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(self.width * scale, out_planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1),
                nn.BatchNorm1d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        
        # Split channels
        spx = torch.split(out, self.width, 1)
        
        # Processing each scale
        sp = spx[0] # First part goes straight through (conceptually, though we usually process from i=0 in loop logic if adapting)
        # Standard Res2Net implementation logic:
        # y[i] = conv(x[i]) if i==0; else conv(x[i] + y[i-1])
        
        # However, the loop above in my previous attempt was slightly off for standard Res2Net.
        # Let's stick to a robust implementation:
        
        output_branches = []
        
        # spx[0] is y_0? Usually Res2Net: y1 = x1; y2 = K2(x2+y1); y3 = K3(x3+y2)...
        # Let's align with the standard logic:
        
        y = spx[0]
        output_branches.append(y)
        
        for i in range(self.nums):
            if i == 0:
                # First processed branch (second actual branch) takes x[1] + x[0] (if cascading)?
                # Standard: y[i] = K[i](x[i] + y[i-1])
                # But spx index 0 is "x1".
                # My self.convs[i] corresponds to branch i+1 (since branch 0 has no conv).
                # Let's restart logic to be clear:
                
                # x split into s subsets: x_1, x_2, ..., x_s
                # y_1 = x_1
                # y_2 = K_2(x_2 + y_1)
                # y_3 = K_3(x_3 + y_2)
                pass

        # Re-implementing forward loop cleanly
        out_list = []
        sp = spx[0] # y_1
        out_list.append(sp)
        
        for i in range(self.nums):
            # input to this conv is x_{i+2} + y_{i+1}
            # Wait, spx has `scale` parts.
            # i goes from 0 to scale-2.
            # i=0 corresponds to processing for y_2. Input is x_2 + y_1.
            # spx[i+1] is x_{i+2}.
            
            sp = sp + spx[i+1]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out_list.append(sp)
            
        out = torch.cat(out_list, dim=1)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for 1D"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        b, c, _ = x.size()
        # Squeeze: Global Average Pooling
        y = x.mean(dim=2) 
        # Excitation
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y

@register_backend("ECAPA_TDNN")
class ECAPA_TDNN(BaseBackend):
    """
    ECAPA-TDNN implementation for ASV Spoofing.
    Inputs: Features [B, T, C] (from SSL models usually) -> we transpose to [B, C, T].
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Input channels (from frontend, e.g. 1024 for Wav2Vec2 Large)
        in_channels = self.input_dim 
        
        # ECAPA Config
        channels = config.get("channels", 512)
        emb_dim = config.get("emb_dim", 192)
        
        # 1. Initial Conv1d
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels),
        )
        
        # 2. SE-Res2Blocks (3 layers usually)
        self.layer2 = Res2NetBlock(channels, channels, scale=8)
        self.layer3 = Res2NetBlock(channels, channels, scale=8)
        self.layer4 = Res2NetBlock(channels, channels, scale=8)
        
        # 3. Multi-scale feature aggregation (MFA)
        # We concatenate the outputs of layer 2, 3, 4
        self.conv_cat = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, kernel_size=1), # 1x1 Conv
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels * 3)
        )
        
        # 4. Attentive Statistics Pooling
        # ASP layer outputs 2*C (mean+std)
        self.pool_layer = pooling_modules.get_pooling_layer(
            {"pooling_type": "asp", "att_hidden_size": 128}, 
            input_dim=channels*3
        )
        self.pooled_dim = self.pool_layer.get_output_dim()
        
        # 5. Final BN and Linear Projection
        self.bn = nn.BatchNorm1d(self.pooled_dim)
        self.linear = nn.Linear(self.pooled_dim, emb_dim)
        
        # Output dimension for the Loss
        self.output_dim = emb_dim

    def forward(self, x, **kwargs):
        """
        x: [B, T, C] (Standard SSL output)
        """
        # Transpose to [B, C, T] for Conv1d
        x = x.transpose(1, 2)
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        
        # Feature Aggregation
        out = torch.cat([out2, out3, out4], dim=1) # [B, 3*C, T]
        out = self.conv_cat(out)
        
        stats = self.pool_layer(out) # [B, pooled_dim]
        
        stats = self.bn(stats)
        embedding = self.linear(stats)
        
        return embedding
