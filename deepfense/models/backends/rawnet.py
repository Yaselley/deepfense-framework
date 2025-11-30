import torch
import torch.nn as nn
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend

class RawNet2Block(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels) if not first else nn.Identity()
        self.act = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()
            
        self.mp = nn.MaxPool1d(3)
        self.first = first

    def forward(self, x):
        # x: [B, C, T]
        res = self.downsample(x)
        
        if not self.first:
            out = self.bn1(x)
            out = self.act(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        
        out += res
        out = self.mp(out)
        return out

@register_backend("RawNet2")
class RawNet2(BaseBackend):
    """
    Modified RawNet2 backend. 
    Typically operates on Raw Audio, but can be adapted for Features if SincNet is skipped.
    Here we implement the 'Back-end' part (ResBlocks + GRU + FC).
    """
    def __init__(self, config):
        super().__init__(config)
        
        in_dim = self.input_dim
        # If input is Wav2Vec (1024), we project it down or use it as channels?
        # RawNet usually takes ~sincnet outputs. 
        # Let's treat 'channels' as the feature dim.
        
        # Config
        self.filts = config.get("filts", [128, 128, 256, 256, 256, 256])
        self.gru_node = config.get("gru_node", 1024)
        self.emb_dim = config.get("emb_dim", 128)
        self.output_dim = self.emb_dim
        
        # Initial projection to match first filter count if needed
        self.first_conv = nn.Conv1d(in_dim, self.filts[0], 1)
        
        self.blocks = nn.ModuleList()
        current_dim = self.filts[0]
        
        for i, f_out in enumerate(self.filts):
            # Note: RawNet blocks usually include MaxPool(3)
            # This aggressively reduces time dimension.
            self.blocks.append(RawNet2Block(current_dim, f_out, first=(i==0)))
            current_dim = f_out
            
        self.bn_before_gru = nn.BatchNorm1d(current_dim)
        self.act = nn.LeakyReLU(0.3)
        
        self.gru = nn.GRU(input_size=current_dim, hidden_size=self.gru_node, 
                          num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(self.gru_node, self.emb_dim)
        
    def forward(self, x, **kwargs):
        # x: [B, T, C] -> Transpose to [B, C, T]
        x = x.transpose(1, 2)
        
        x = self.first_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.bn_before_gru(x)
        x = self.act(x)
        
        # GRU expects [B, T, C]
        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        _, hn = self.gru(x) # hn: [1, B, Hidden]
        
        x = hn[-1] # [B, Hidden]
        embedding = self.fc(x)
        
        return embedding
