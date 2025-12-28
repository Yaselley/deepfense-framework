# Adding a New Learning Rate Scheduler

This guide shows you how to add a custom learning rate scheduler to DeepFense.

## Overview

Schedulers in DeepFense adjust the learning rate during training. They are registered with `@register_scheduler` and wrap PyTorch's learning rate schedulers. Schedulers are functions that take an optimizer and config, returning a scheduler instance.

## Step-by-Step Guide

### Step 1: Add to Scheduler Registry

Add your scheduler to `deepfense/training/schedulers/utils.py`:

```python
import torch
from deepfense.utils.registry import register_scheduler


@register_scheduler("my_scheduler")
def MyScheduler(optimizer, config):
    """
    Custom learning rate scheduler builder function.
    
    Args:
        optimizer: PyTorch optimizer instance
        config: Dictionary containing scheduler configuration
            - step_size: Step size for scheduler
            - gamma: Decay factor
            - Other scheduler-specific parameters
    
    Returns:
        PyTorch scheduler instance
    """
    step_size = config.get("step_size", 30)
    gamma = config.get("gamma", 0.1)
    
    # Create and return scheduler
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
```

### Step 2: Verify Registration

The scheduler is automatically registered when the module is imported. Check that it's registered:

```bash
deepfense list  # Schedulers might not be in the list, check code directly
```

Or programmatically:

```python
from deepfense.training.schedulers import utils  # Import to register
from deepfense.utils.registry import SCHEDULER_REGISTRY

# Check if registered
if "my_scheduler" in SCHEDULER_REGISTRY:
    print("Scheduler registered successfully!")
    print("Available schedulers:", SCHEDULER_REGISTRY.list())
```

### Step 3: Use in Configuration

Use your scheduler in a YAML configuration file:

```yaml
training:
  optimizer:
    type: "Adam"
    lr: 0.001
  scheduler:
    type: "my_scheduler"  # Your registered name
    step_size: 30
    gamma: 0.1
```

## Complete Example: Cosine Annealing Scheduler

Here's a complete example for Cosine Annealing:

```python
import torch
from deepfense.utils.registry import register_scheduler


@register_scheduler("CosineAnnealingLR")
def CosineAnnealingScheduler(optimizer, config):
    """
    Cosine annealing learning rate scheduler.
    
    Linearly decreases LR following cosine function.
    """
    T_max = config.get("T_max", 50)  # Maximum number of iterations
    eta_min = config.get("eta_min", 0)  # Minimum learning rate
    last_epoch = config.get("last_epoch", -1)
    
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch
    )
```

## Example: Exponential Decay Scheduler

```python
import torch
from deepfense.utils.registry import register_scheduler


@register_scheduler("ExponentialLR")
def ExponentialScheduler(optimizer, config):
    """
    Exponential decay learning rate scheduler.
    
    Multiplies LR by gamma every epoch.
    """
    gamma = config.get("gamma", 0.95)
    last_epoch = config.get("last_epoch", -1)
    
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma,
        last_epoch=last_epoch
    )
```

## Example: ReduceLROnPlateau Scheduler

```python
import torch
from deepfense.utils.registry import register_scheduler


@register_scheduler("ReduceLROnPlateau")
def ReduceLROnPlateauScheduler(optimizer, config):
    """
    Reduce learning rate when a metric has stopped improving.
    
    Monitors a metric and reduces LR when it plateaus.
    """
    mode = config.get("mode", "min")  # 'min' or 'max'
    factor = config.get("factor", 0.1)  # Factor by which LR is reduced
    patience = config.get("patience", 10)  # Number of epochs to wait
    threshold = config.get("threshold", 1e-4)
    threshold_mode = config.get("threshold_mode", "rel")
    cooldown = config.get("cooldown", 0)
    min_lr = config.get("min_lr", 0)
    eps = config.get("eps", 1e-8)
    
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps
    )
```

## Example: Warmup + Cosine Annealing

Here's an example of a custom scheduler combining warmup with cosine annealing:

```python
import torch
from torch.optim.lr_scheduler import _LRScheduler
from deepfense.utils.registry import register_scheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup: linear increase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            t = self.last_epoch - self.warmup_epochs
            T = self.T_max - self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + torch.cos(torch.tensor(t * 3.14159 / T))) / 2
                for base_lr in self.base_lrs
            ]


@register_scheduler("WarmupCosine")
def WarmupCosineSchedulerBuilder(optimizer, config):
    """
    Builder for warmup + cosine annealing scheduler.
    """
    warmup_epochs = config.get("warmup_epochs", 5)
    T_max = config.get("T_max", 50)
    eta_min = config.get("eta_min", 0)
    
    return WarmupCosineScheduler(optimizer, warmup_epochs, T_max, eta_min)
```

## Key Points

1. **Use @register_scheduler decorator**: Register with a unique string name
2. **Function signature**: Must accept `(optimizer, config)` where `optimizer` is a PyTorch optimizer
3. **Return scheduler**: Must return a PyTorch scheduler instance
4. **Config parameters**: Extract parameters from the config dictionary
5. **Default values**: Provide sensible defaults for optional parameters
6. **No import needed**: Schedulers are registered when the module is imported

## Function Signature

Your scheduler function should follow this pattern:

```python
@register_scheduler("scheduler_name")
def MyScheduler(optimizer, config):
    """
    Args:
        optimizer: PyTorch optimizer instance
        config: Dictionary with scheduler configuration
            - Parameter1: Description
            - Parameter2: Description
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: PyTorch scheduler instance
    """
    param1 = config.get("param1", default_value)
    # ... other parameters
    
    return torch.optim.lr_scheduler.SomeScheduler(optimizer, param1=param1, ...)
```

## Testing Your Scheduler

Test your scheduler before using it in training:

```python
import torch
import torch.nn as nn
from deepfense.training.schedulers import utils  # Import to register
from deepfense.utils.registry import build_scheduler

# Create a simple model and optimizer
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create scheduler config
scheduler_config = {
    "type": "my_scheduler",
    "step_size": 10,
    "gamma": 0.5
}

# Build scheduler
scheduler = build_scheduler(scheduler_config["type"], optimizer, scheduler_config)

# Test scheduler step
print(f"Initial LR: {optimizer.param_groups[0]['lr']}")
for epoch in range(5):
    scheduler.step()
    print(f"Epoch {epoch+1} LR: {optimizer.param_groups[0]['lr']}")
```

## Available PyTorch Schedulers

DeepFense supports any PyTorch scheduler. Common ones include:

- `torch.optim.lr_scheduler.StepLR` - Step decay
- `torch.optim.lr_scheduler.ExponentialLR` - Exponential decay
- `torch.optim.lr_scheduler.CosineAnnealingLR` - Cosine annealing
- `torch.optim.lr_scheduler.ReduceLROnPlateau` - Reduce on plateau
- `torch.optim.lr_scheduler.MultiStepLR` - Multi-step decay
- `torch.optim.lr_scheduler.OneCycleLR` - One cycle policy

## Next Steps

- See [Adding Optimizers](adding_optimizers.md) for optimizer creation
- See [Training Guide](training_with_cli.md) for how to use schedulers in training
- See [Configuration Reference](../05_configuration.md) for full config options
- See existing schedulers in `deepfense/training/schedulers/utils.py` for reference

