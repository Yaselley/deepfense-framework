import torch
from deepfense.training.optimizers.registry import register_optimizer

@register_optimizer("adam")
def AdamOptimizer(params, lr=1e-6, weight_decay=1e-04):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

@register_optimizer("adamw")
def AdamWOptimizer(params, lr=1e-6, weight_decay=1e-04):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

@register_optimizer("sgd")
def SGDOptimizer(params, lr=1e-6, momentum=0.9, weight_decay=1e-04):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
