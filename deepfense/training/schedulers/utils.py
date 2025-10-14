import torch
from deepfense.training.schedulers.registry import register_scheduler

@register_scheduler("step")
def StepLRScheduler(optimizer, step_size=10, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

@register_scheduler("cosine")
def CosineAnnealingLRScheduler(optimizer, T_max=50, eta_min=0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

@register_scheduler("exponential")
def ExponentialLRScheduler(optimizer, gamma=0.95):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

@register_scheduler("reduce_on_plateau")
def ReduceLROnPlateauScheduler(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                                      patience=patience, threshold=threshold)