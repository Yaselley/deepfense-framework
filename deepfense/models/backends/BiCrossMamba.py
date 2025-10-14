import torch.nn as nn
import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
import math
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                     
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: B*C*D*T
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Residual_block(nn.Module):
    def __init__(self, nb_filts, pool, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        
        self.mp = nn.MaxPool2d(pool)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        pool = config.get("pool", (1, 1))

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], pool=pool, first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2], pool=pool)),
            nn.Sequential(Residual_block(nb_filts=filts[3], pool=pool)),
            nn.Sequential(Residual_block(nb_filts=filts[4], pool=pool)),
            nn.Sequential(Residual_block(nb_filts=filts[4], pool=pool)),
            nn.Sequential(Residual_block(nb_filts=filts[4], pool=pool)))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.max_pool2d(torch.abs(x), (3, 3)) 
        x = self.first_bn(x) 
        x = self.selu(x) 
        x = self.encoder(x) 
        x = self.first_bn1(x)
        x = self.selu(x)

        return x

class AttentionMap(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.selu = nn.SELU()
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.selu(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.squeeze(1)
        attention_map = F.softmax(x.view(x.size(0), -1), dim=-1).view(x.size(0), x.size(1), x.size(2))
        return attention_map

class MutualCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(MutualCrossAttention, self).__init__()
        # Define two linear layers for the weight matrices
        self.head_attn_1 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.head_attn_2 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, xF, xT):
        """
        xF: Tensor of shape (C, B, T)
        xT: Tensor of shape (C, H, T)
        Returns:
            xF_prime: Updated xF with attention from xT
            xT_prime: Updated xT with attention from xF
        """
        xF_attn, _ = self.head_attn_1(xF, xT, xT)
        xT_attn, _ = self.head_attn_2(xT, xF, xF)

        xF = xF + xF_attn
        xT = xT + xT_attn

        xF = self.layer_norm_1(xF)
        xT = self.layer_norm_2(xT)

        return xF, xT

class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.d_conv = 4
        self.d_state = d_state
        self.layer_idx = layer_idx
        self.residual_in_fp32 = True
        self.norm = MambaRMSNorm(self.d_model, eps=1e-05)
        
        self.norm_backward = MambaRMSNorm(self.d_model, eps=1e-05)
        self.norm_forward = MambaRMSNorm(self.d_model, eps=1e-05)

        self.forward_mamba = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv
        )
        self.backward_mamba = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv
        )

        self.output_proj = nn.Linear(self.d_model*2, self.d_model)

    def forward(
        self,
        hidden_states,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states_fd = self.forward_mamba(
            hidden_states
        )
        hidden_states_bk = self.backward_mamba(
            hidden_states.flip([1])
        )
        
        hidden_states_fd = residual + hidden_states_fd
        hidden_states_bk = residual.flip([1]) + hidden_states_bk

        hidden_states_fd = self.norm_forward(hidden_states_fd)
        hidden_states_bk = self.norm_backward(hidden_states_bk.flip([1]))

        hidden_states = torch.cat((hidden_states_fd, hidden_states_bk), dim=-1)
        hidden_states = self.output_proj(hidden_states)

        return hidden_states

class BiCrossMamba_ST(nn.Module):
    def __init__(
        self,
        config=None,
    ) -> None:
        self.config = config
        self.d_model = 64
        self.n_layer = self.config.get("n_layers", 3) 
        self.h_channels = 1
        self.d_state = 16

        super().__init__()

        self.encoder = Encoder(config)
        self.attention_map = AttentionMap(hidden_channels=1)

        self.mamba_t = nn.ModuleList([BiMambaBlock(d_model=self.d_model, d_state=self.d_state) for i in range(self.n_layer)])
        self.mamba_f = nn.ModuleList([BiMambaBlock(d_model=self.d_model, d_state=self.d_state) for i in range(self.n_layer)])

        self.crossattn = MutualCrossAttention(embed_dim=self.d_model)

        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)

        self.selu = nn.SELU()
        self.norm_f = MambaRMSNorm(self.d_model)
        self.norm_t = MambaRMSNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.encoder(x)

        x_attn = self.attention_map(x)
        x_attn = x_attn.unsqueeze(1)

        xt = (x_attn*x).sum(dim=-1)
        xf = (x_attn*x).sum(dim=-2)

        xt = xt.permute(0,2,1)
        xf = xf.permute(0,2,1)

        for idx, ly in enumerate(self.mamba_t):
            xt = ly(xt)
            xf = self.mamba_f[idx](xf)

        xt, xf = self.crossattn(xt, xf)

        xt = self.norm_t(xt)
        xf = self.norm_f(xf)

        xt = torch.matmul(F.softmax(self.fc1(xt), dim=1).transpose(-1, -2), xt).squeeze(-2)
        xf = torch.matmul(F.softmax(self.fc2(xf), dim=1).transpose(-1, -2), xf).squeeze(-2)

        x = torch.cat((xt, xf), dim=-1)

        return x
