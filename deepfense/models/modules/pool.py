import torch
import torch.nn as nn
import torch.nn.functional as F

class TdnnAffine(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        context=[0],
        bias=True,
        pad=True,
        stride=1,
        groups=1,
    ):
        super(TdnnAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.pad = pad
        self.stride = stride
        self.groups = groups
        self.left_context = context[0] if context[0] < 0 else 0
        self.right_context = context[-1] if context[-1] > 0 else 0
        self.tot_context = self.right_context - self.left_context + 1

        self.weight = nn.Parameter(
            torch.randn(output_dim, input_dim // groups, self.tot_context)
        )
        self.bias = nn.Parameter(torch.randn(output_dim)) if bias else None
        nn.init.kaiming_normal_(self.weight)

        # Mask creation for context logic
        if len(context) != self.tot_context:
            self.mask = torch.zeros(1, 1, self.tot_context)
            for idx, c in enumerate(range(self.left_context, self.right_context + 1)):
                if c in context:
                    self.mask[0, 0, idx] = 1
        else:
            self.mask = None

    def forward(self, x):
        # x: [B, C, T]
        if self.pad:
            x = F.pad(
                x, (-self.left_context, self.right_context), mode="constant", value=0.0
            )

        if self.mask is not None:
            self.mask = self.mask.to(x.device)
            w = self.weight * self.mask
        else:
            w = self.weight

        return F.conv1d(x, w, self.bias, stride=self.stride, groups=self.groups)


class AttentionAlphaComponent(nn.Module):
    def __init__(
        self,
        input_dim,
        num_head=1,
        split_input=True,
        share=True,
        affine_layers=2,
        hidden_size=64,
        context=[0],
        bias=True,
    ):
        super(AttentionAlphaComponent, self).__init__()
        self.input_dim = input_dim
        if num_head > 1 and split_input:
            assert input_dim % num_head == 0

        if share:
            final_dim = 1
        elif split_input:
            final_dim = input_dim // num_head
        else:
            final_dim = input_dim

        self.relu_affine = affine_layers == 2
        if self.relu_affine:
            self.first_affine = TdnnAffine(
                input_dim,
                hidden_size * num_head,
                context=context,
                bias=bias,
                groups=num_head if (num_head > 1 and split_input) else 1,
            )
            self.relu = nn.ReLU(inplace=True)
            last_in_dim = hidden_size * num_head
        else:
            last_in_dim = input_dim

        self.last_affine = TdnnAffine(
            last_in_dim,
            final_dim * num_head,
            context=context,
            bias=bias,
            groups=num_head if (num_head > 1 and split_input) else 1,
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        if self.relu_affine:
            x = self.relu(self.first_affine(x))
        return self.softmax(self.last_affine(x))


class TAP(nn.Module):
    """Temporal Average Pooling (Standard Mean)"""

    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim
        self.output_dim = in_dim

    def forward(self, x):
        # x: [B, C, T]
        return x.mean(dim=-1)

    def get_output_dim(self):
        return self.output_dim


class StatisticsPooling(nn.Module):
    """Mean + StdDev Pooling"""

    def __init__(self, input_dim, stddev=True, eps=1e-5, **kwargs):
        super(StatisticsPooling, self).__init__()
        self.stddev = stddev
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim if stddev else input_dim
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=2)
        if self.stddev:
            std = torch.sqrt((x.var(dim=2) + self.eps))
            return torch.cat((mean, std), dim=1)
        return mean

    def get_output_dim(self):
        return self.output_dim


class AttentiveStatisticsPooling(nn.Module):
    """Weighted Mean + StdDev based on Attention"""

    def __init__(
        self,
        input_dim,
        affine_layers=2,
        hidden_size=64,
        context=[0],
        stddev=True,
        **kwargs,
    ):
        super(AttentiveStatisticsPooling, self).__init__()
        self.stddev = stddev
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim if stddev else input_dim
        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=1,
            share=True,
            affine_layers=affine_layers,
            hidden_size=hidden_size,
            context=context,
        )

    def forward(self, x):
        # x: [B, C, T]
        alpha = self.attention(x)  # [B, 1, T]
        mean = torch.sum(alpha * x, dim=2)
        if self.stddev:
            var = torch.sum(alpha * x**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-10))
            return torch.cat((mean, std), dim=1)
        return mean

    def get_output_dim(self):
        return self.output_dim


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling (ASP with split heads)"""

    def __init__(self, input_dim, num_head=4, stddev=True, affine_layers=1, **kwargs):
        super(MultiHeadAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.output_dim = 2 * input_dim if stddev else input_dim
        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=num_head,
            split_input=True,
            share=True,
            affine_layers=affine_layers,
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        alpha = self.attention(x)  # Weights

        # Reshape for multi-head
        # Split features into heads: [B, Head, Feat/Head, T]
        x_reshaped = x.reshape(B, self.num_head, -1, T)
        alpha_reshaped = alpha.reshape(B, self.num_head, -1, T)

        mean = torch.sum(alpha_reshaped * x_reshaped, dim=3)  # Sum over T
        mean = mean.reshape(B, -1)  # Flatten heads back to C

        if self.stddev:
            var = torch.sum(alpha_reshaped * x_reshaped**2, dim=3)
            var = var.reshape(B, -1) - mean**2
            std = torch.sqrt(var.clamp(min=1e-10))
            return torch.cat((mean, std), dim=1)
        return mean

    def get_output_dim(self):
        return self.output_dim



def get_pooling_layer(config, input_dim):
    """
    Selects the pooling layer based on config.pooling_type
    """
    ptype = config.get("pooling_type", "mean").lower()
    stddev = config.get("stddev", True)  # Use stats/std by default if applicable

    if ptype in ["tap", "mean"]:
        return TAP(input_dim)

    elif ptype in ["stats", "statistics"]:
        return StatisticsPooling(input_dim, stddev=stddev)

    elif ptype in ["att_stats", "asp"]:
        return AttentiveStatisticsPooling(
            input_dim, stddev=stddev, hidden_size=config.get("att_hidden_size", 64)
        )

    elif ptype in ["mha", "multihead"]:
        return MultiHeadAttentionPooling(
            input_dim, stddev=stddev, num_head=config.get("heads", 4)
        )

    else:
        raise ValueError(f"Unknown pooling type: {ptype}")
