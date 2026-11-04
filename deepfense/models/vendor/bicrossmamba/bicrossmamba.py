import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mamba_ssm import Mamba


class AngleLinear(nn.Module):
    """Angular margin linear layer (A-Softmax / SphereFace)."""

    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def forward(self, input):
        x = input
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / math.pi).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = self.mlambda[self.m](cos_theta)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        return (cos_theta, phi_theta)


class AngleLoss(nn.Module):
    """A-Softmax loss for use with AngleLinear."""

    def __init__(self, gamma=0, lambda_min=5.0, lambda_max=1500.0):
        super().__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lamb = lambda_max

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.long(), 1)
        index = index.bool()

        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.01 * self.it))
        output = cos_theta * 1.0
        output[index] -= cos_theta[index] * (1.0 / (1 + self.lamb))
        output[index] += phi_theta[index] * (1.0 / (1 + self.lamb))

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target.long())
        logpt = logpt.view(-1)

        loss = -logpt.mean()
        return loss


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0], out_channels=nb_filts[1],
            kernel_size=(2, 3), padding=(1, 1), stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1], out_channels=nb_filts[1],
            kernel_size=(2, 3), padding=(0, 1), stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0], out_channels=nb_filts[1],
                padding=(0, 1), kernel_size=(1, 3), stride=1,
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class RawNet_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)

        self.proj = nn.Linear(config.get("input_dim", 1024), 128)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

    def forward(self, x):
        x = self.proj(x)
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
        attention_map = F.softmax(
            x.view(x.size(0), -1), dim=-1
        ).view(x.size(0), x.size(1), x.size(2))
        return attention_map


class MutualCrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.head_attn_1 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.head_attn_2 = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, xF, xT):
        xF_attn, _ = self.head_attn_1(xF, xT, xT)
        xT_attn, _ = self.head_attn_2(xT, xF, xF)

        xF = self.layer_norm_1(xF + xF_attn)
        xT = self.layer_norm_2(xT + xT_attn)
        return xF, xT


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
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
    """Bidirectional Mamba block with forward and backward SSM passes."""

    def __init__(self, d_model, d_state, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.d_conv = 4
        self.d_state = d_state
        self.residual_in_fp32 = True

        self.norm = MambaRMSNorm(d_model, eps=1e-05)
        self.norm_forward = MambaRMSNorm(d_model, eps=1e-05)
        self.norm_backward = MambaRMSNorm(d_model, eps=1e-05)

        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=self.d_conv)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=self.d_conv)

        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states_fd = self.forward_mamba(hidden_states)
        hidden_states_bk = self.backward_mamba(hidden_states.flip([1]))

        hidden_states_fd = residual + hidden_states_fd
        hidden_states_bk = residual.flip([1]) + hidden_states_bk

        hidden_states_fd = self.norm_forward(hidden_states_fd)
        hidden_states_bk = self.norm_backward(hidden_states_bk.flip([1]))

        hidden_states = torch.cat((hidden_states_fd, hidden_states_bk), dim=-1)
        hidden_states = self.output_proj(hidden_states)
        return hidden_states


class BiCrossMamba_ST(nn.Module):
    """
    BiCrossMamba-ST: Bidirectional Cross-Mamba with Spectral-Temporal processing
    for synthetic speech detection.

    Pipeline:
        SSL features → RawNet encoder → Attention Map →
        → Spectral/Temporal BiMamba → Mutual Cross-Attention →
        → Pooling → AngleLinear classifier
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.d_model = 64
        self.n_layer = self.config.get("n_layers", 3)
        self.d_state = 16

        super().__init__()

        self.encoder = RawNet_Encoder(self.config)
        self.attention_map = AttentionMap(hidden_channels=1)

        self.mamba_t = nn.ModuleList([
            BiMambaBlock(d_model=self.d_model, d_state=self.d_state)
            for _ in range(self.n_layer)
        ])
        self.mamba_f = nn.ModuleList([
            BiMambaBlock(d_model=self.d_model, d_state=self.d_state)
            for _ in range(self.n_layer)
        ])

        self.crossattn = MutualCrossAttention(embed_dim=self.d_model)

        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)

        self.proj_ATTN = AngleLinear(128, 2)

        self.selu = nn.SELU()
        self.norm_f = MambaRMSNorm(self.d_model)
        self.norm_t = MambaRMSNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.forward_from_map(self.encoder(x))

    def forward_from_map(self, x):
        x_attn = self.attention_map(x)
        x_attn = x_attn.unsqueeze(1)

        xt = (x_attn * x).sum(dim=-1)
        xf = (x_attn * x).sum(dim=-2)

        xt = xt.permute(0, 2, 1)
        xf = xf.permute(0, 2, 1)

        for idx, ly in enumerate(self.mamba_t):
            xt = ly(xt)
            xf = self.mamba_f[idx](xf)

        xt, xf = self.crossattn(xt, xf)

        xt = self.norm_t(xt)
        xf = self.norm_f(xf)

        xt = torch.matmul(
            F.softmax(self.fc1(xt), dim=1).transpose(-1, -2), xt
        ).squeeze(-2)
        xf = torch.matmul(
            F.softmax(self.fc2(xf), dim=1).transpose(-1, -2), xf
        ).squeeze(-2)

        x = torch.cat((xt, xf), dim=-1)
        x = self.proj_ATTN(x)
        return x
