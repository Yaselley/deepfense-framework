import torch
from deepfense.utils.registry import register_scheduler


@register_scheduler("step_lr")
def StepLRScheduler(optimizer, config):
    step_size = config.get("step_size", 10)
    gamma = config.get("gamma", 0.1)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("cosine")
def CosineAnnealingLRScheduler(optimizer, config):
    T_max = config.get("T_max", 50)
    eta_min = config.get("eta_min", 0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )


@register_scheduler("exponential")
def ExponentialLRScheduler(optimizer, config):
    gamma = config.get("gamma", 0.9)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
