#!/usr/bin/env python
"""
a_softmax layers

copied from https://github.com/Joyako/SphereFace-pytorch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from deepfense.utils.registry import register_loss


class AngleLayer(nn.Module):
    """Output layer to produce activation for Angular softmax layer"""

    def __init__(self, config):
        super(AngleLayer, self).__init__()

        in_planes = config["embedding_dim"]
        out_planes = config["n_classes"]
        m = config["m"]

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        # cos(m \theta) = f(cos(\theta))
        self.cos_val = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

    def forward(self, input, flag_angle_only=False):
        """
        Compute a-softmax activations
        """
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1) / w_modulus.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        # cos(m \theta)
        cos_m_theta = self.cos_val[self.m](cos_theta)

        with torch.no_grad():
            theta = cos_theta.acos()
            k = (self.m * theta / 3.14159265).floor()
            minus_one = k * 0.0 - 1

        phi_theta = (minus_one**k) * cos_m_theta - 2 * k

        if flag_angle_only:
            cos_x = cos_theta
            phi_x = phi_theta
        else:
            cos_x = cos_theta * x_modulus.view(-1, 1)
            phi_x = phi_theta * x_modulus.view(-1, 1)

        return cos_x, phi_x


from deepfense.models.base_model import BaseLoss

@register_loss("ASoftmax")
class ASoftmaxLoss(BaseLoss):
    """
    Unified Angular Softmax Loss.
    Includes the AngleLayer projection.
    """

    def __init__(self, config):
        super().__init__(config)

        self.mapper = AngleLayer(config)

        self.gamma = config.get("gamma", 0.5)
        self.iter = config.get("iter", 0)
        self.lambda_min = config.get("lambda_min", 5)
        self.lambda_max = config.get("lambda_max", 1500)
        self.lamb = config.get("lamb", 1500)

    def forward(self, embeddings, target):
        """
        embeddings: (batchsize, embedding_dim)
        target: (batchsize)
        """
        # Calculate cos_x and phi_x
        cos_x, phi_x = self.mapper(embeddings)
        input_tuple = (cos_x, phi_x)

        self.iter += 1
        target = target.long().view(-1, 1)

        with torch.no_grad():
            index = torch.zeros_like(input_tuple[0])
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.bool()

        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input_tuple[0] * 1.0
        output[index] -= input_tuple[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input_tuple[1][index] * 1.0 / (1 + self.lamb)

        logit = F.log_softmax(output, dim=1)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()
        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss

    def get_score(self, embeddings):
        """
        Returns final scores for validation/inference.
        If n_classes == 2, returns (cos_bonafide - cos_spoof).
        Otherwise returns full cosine scores.
        """
        cos_x = self.get_logits(embeddings)
        if cos_x.shape[1] == 2:
             # Return difference based on configured labels
             return cos_x[:, self.bonafide_label] - cos_x[:, self.spoof_label]
        return cos_x

    def get_logits(self, embeddings):
        """Returns full cosine scores [N, C] for caching/loss."""
        cos_x, _ = self.mapper(embeddings)
        return cos_x
