import torch
from deepfense.training.optimizers.registry import register_optimizer


@register_optimizer("adam")
def AdamOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    weight_decay = config.get("weight_decay", 1e-04)
    betas = config.get("betas", (0.9, 0.999))
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)


@register_optimizer("adamw")
def AdamWOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    weight_decay = config.get("weight_decay", 1e-04)
    betas = config.get("betas", (0.9, 0.999))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


@register_optimizer("sgd")
def SGDOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-04)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
