import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from deepfense.models.backends.registry import register_backend

@register_backend("linear_layer")
class LR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(config["input_dim"], config["num_classes"])

    def forward(self, x):
    
        features = self.linear(x.mean(dim=1))
        return features