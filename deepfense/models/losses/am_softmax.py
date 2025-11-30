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

        # ((batchsize, output_dim), (batchsie, output_dim))
        return cos_x, phi_x


@register_loss("AMSoftmax")
class AMSoftmaxLoss(nn.Module):
    """
    Unified AMSoftmax Loss + AngleLayer.
    """

    def __init__(self, config):
        super(AMSoftmaxLoss, self).__init__()

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
            # In AMSoftmax, get_logits returns cos_x, but we need (cos_x, phi_x) for training.
            # If the passed 'logits' is just cos_x (from get_logits), we still need phi_x.
            # So we might still need to run mapper if logits doesn't contain both.
            # However, StandardDetector only gets 'scores' via get_logits() which returns cos_x.
            # Re-running mapper is safer here unless we change get_logits to return full tuple.
            # For now, we will NOT use the cached logits for AMSoftmax to avoid correctness issues,
            # unless we change get_logits to return the full tuple (which we probably shouldn't for validation).
            pass
            
        input_tuple = self.mapper(embeddings)
        
        # target (batchsize)
        target = target.long()

        # create an index matrix, i.e., one-hot vectors
        with torch.no_grad():
            index = torch.zeros_like(input_tuple[0])
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.bool()

        # use the one-hot vector as index to select
        # input[0] -> cos
        # input[1] -> phi
        # if target_i = j, ouput[i][j] = phi[i][j], otherwise cos[i][j]
        #
        output = input_tuple[0] * 1.0
        output[index] -= input_tuple[0][index] * 1.0
        output[index] += input_tuple[1][index] * 1.0

        # cross entropy loss
        loss = self.m_loss(output, target)

        return loss

    def get_logits(self, embeddings):
        """Returns cos_x as logits."""
        cos_x, _ = self.mapper(embeddings)
        return cos_x
