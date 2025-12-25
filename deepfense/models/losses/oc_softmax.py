"""
one class

One-class learning towards generalized voice spoofing detection
Zhang, You and Jiang, Fei and Duan, Zhiyao
arXiv preprint arXiv:2010.13995
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from deepfense.utils.registry import register_loss


class OCAngleLayer(nn.Module):
    """Output layer to produce activation for one-class softmax"""

    def __init__(self, config):
        super(OCAngleLayer, self).__init__()
        in_planes = config["embedding_dim"]
        w_posi = config["w_posi"]
        w_nega = config["w_nega"]
        alpha = config["alpha"]

        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1

        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations
        """
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)

        return pos_score, neg_score


from deepfense.models.base_model import BaseLoss

@register_loss("OCSoftmax")
class OCSoftmaxLoss(BaseLoss):
    """
    Unified OCSoftmax Loss + AngleLayer.
    """

    def __init__(self, config):
        super().__init__(config)
        if "in_planes" not in config and "embedding_dim" in config:
             config["in_planes"] = config["embedding_dim"]
             
        self.mapper = OCAngleLayer(config)
        self.m_loss = nn.Softplus()

    def forward(self, embeddings, target, logits=None):
        """
        embeddings: (batch, dim)
        target: (batch,)
        logits: Optional pre-computed inputs tuple.
        """
        if logits is not None and isinstance(logits, tuple):
             inputs = logits
        else:
             inputs = self.mapper(embeddings)
        
        output = inputs[0] * target.view(-1, 1) + inputs[1] * (1 - target.view(-1, 1))
        loss = self.m_loss(output).mean()

        return loss

    def get_score(self, embeddings):
        """
        Returns cos_theta (similarity to bonafide center).
        """
        cos_theta = self.get_logits(embeddings)
        return cos_theta.squeeze(1)

    def get_logits(self, embeddings):
        """Returns full cos_theta [N, 1] for caching/loss."""
        cos_theta, _ = self.mapper(embeddings, flag_angle_only=True)
        return cos_theta
