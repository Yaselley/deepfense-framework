#!/usr/bin/env python
"""
additive margin softmax layers

Wang, F., Cheng, J., Liu, W. & Liu, H.
Additive margin softmax for face verification. IEEE Signal Process. Lett. 2018

"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from deepfense.utils.registry import register_loss


class AMAngleLayer(nn.Module):
    """Output layer to produce activation for Angular softmax layer"""

    def __init__(self, config):
        super(AMAngleLayer, self).__init__()

        in_planes = config["embedding_dim"]
        out_planes = config["n_classes"]
        s = config["s"]
        m = config["m"]

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.m = m
        self.s = s

    def forward(self, input, flag_angle_only=False):
        """
        Compute am-softmax activations
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
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        if flag_angle_only:
            cos_x = cos_theta
            phi_x = cos_theta
        else:
            cos_x = self.s * cos_theta
            phi_x = self.s * (cos_theta - self.m)

        return cos_x, phi_x


from deepfense.models.base_model import BaseLoss

@register_loss("AMSoftmax")
class AMSoftmaxLoss(BaseLoss):
    """
    Unified AMSoftmax Loss + AngleLayer.
    """

    def __init__(self, config):
        super().__init__(config)

        self.mapper = AMAngleLayer(config)

        class_weights = config.get("class_weights", [0.5, 0.5])
        reduction = config.get("reduction", "mean")
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        self.m_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def forward(self, embeddings, target, logits=None):
        """
        embeddings: (batch, dim)
        target: (batch,)
        logits: (optional) pre-computed tuple (cos_x, phi_x) from mapper
        """
        if logits is not None:
            pass
            
        input_tuple = self.mapper(embeddings)
        
        target = target.long()

        with torch.no_grad():
            index = torch.zeros_like(input_tuple[0])
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.bool()

        output = input_tuple[0] * 1.0
        output[index] -= input_tuple[0][index] * 1.0
        output[index] += input_tuple[1][index] * 1.0

        loss = self.m_loss(output, target)

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
