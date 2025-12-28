# Adding a New Optimizer

This guide shows you how to add a custom optimizer to DeepFense.

## Overview

Optimizers in DeepFense are registered functions that create PyTorch optimizer instances. They must be registered with `@register_optimizer` in the registry file. Optimizers are typically wrappers around PyTorch's built-in optimizers with custom parameter configurations.

## Step-by-Step Guide

### Step 1: Add to Optimizer Registry

Add your optimizer to `deepfense/training/optimizers/utils.py`:

```python
import torch
from deepfense.utils.registry import register_optimizer


@register_optimizer("my_optimizer")
def MyOptimizer(params, config):
    """
    Custom optimizer builder function.
    
    Args:
        params: Model parameters (iterable) to optimize
        config: Dictionary containing optimizer configuration
            - lr: Learning rate (required)
            - momentum: Momentum parameter (optional)
            - weight_decay: Weight decay (optional)
            - Other optimizer-specific parameters
    
    Returns:
        PyTorch optimizer instance
    """
    lr = config.get("lr", 0.001)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-4)
    nesterov = config.get("nesterov", False)
    
    # Create and return optimizer
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )
```

Alternatively, if you prefer to keep it in the main registry file, add it to `deepfense/utils/registry.py` at the end of the file (before any existing optimizer registrations, or in a dedicated section).

### Step 2: Verify Registration

The optimizer is automatically registered when the module is imported. Check that it's registered:

```bash
deepfense list --component-type optimizers
```

Or programmatically:

```python
from deepfense.training.optimizers import utils  # Import to register
from deepfense.utils.registry import OPTIMIZER_REGISTRY

# Check if registered
if "my_optimizer" in OPTIMIZER_REGISTRY:
    print("Optimizer registered successfully!")
    print("Available optimizers:", OPTIMIZER_REGISTRY.list())
```

### Step 3: Use in Configuration

Use your optimizer in a YAML configuration file:

```yaml
training:
  optimizer:
    type: "my_optimizer"  # Your registered name
    lr: 0.001
    momentum: 0.9
    weight_decay: 1e-4
    nesterov: True
```

## Complete Example: SGD with Custom Settings

Here's a complete example for SGD with custom parameter groups:

```python
import torch
from deepfense.utils.registry import register_optimizer


@register_optimizer("custom_sgd")
def CustomSGDOptimizer(params, config):
    """
    SGD optimizer with separate learning rates for different parameter groups.
    
    Supports:
    - Different learning rates for frontend/backend
    - Layer-wise learning rates
    - Custom momentum and weight decay
    """
    base_lr = config.get("lr", 0.001)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-4)
    nesterov = config.get("nesterov", False)
    
    # Option 1: Simple - same LR for all parameters
    if not config.get("use_param_groups", False):
        return torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    # Option 2: Different LR for different parameter groups
    param_groups = []
    
    # Group 1: Frontend parameters (typically frozen, but if not)
    frontend_lr = config.get("frontend_lr", base_lr * 0.1)
    frontend_params = [p for n, p in params if "frontend" in n]
    if frontend_params:
        param_groups.append({
            "params": frontend_params,
            "lr": frontend_lr,
            "momentum": momentum,
            "weight_decay": weight_decay
        })
    
    # Group 2: Backend parameters
    backend_params = [p for n, p in params if "backend" in n or "loss" in n]
    if backend_params:
        param_groups.append({
            "params": backend_params,
            "lr": base_lr,
            "momentum": momentum,
            "weight_decay": weight_decay
        })
    
    # If no groups matched, use all params
    if not param_groups:
        param_groups = [{"params": list(params)}]
    
    return torch.optim.SGD(param_groups, momentum=momentum, nesterov=nesterov)
```

## Example: Adam with Warmup

Here's an example of Adam optimizer with learning rate warmup (note: warmup is typically handled by schedulers, but shown here for completeness):

```python
import torch
from deepfense.utils.registry import register_optimizer


@register_optimizer("adam_warmup")
def AdamWarmupOptimizer(params, config):
    """
    Adam optimizer with custom beta values.
    """
    lr = config.get("lr", 0.001)
    betas = config.get("betas", (0.9, 0.999))
    weight_decay = config.get("weight_decay", 1e-4)
    eps = config.get("eps", 1e-8)
    amsgrad = config.get("amsgrad", False)
    
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad
    )
```

## Example: RAdam Optimizer

Example of adding RAdam (Rectified Adam) if you have the `timm` library:

```python
import torch
from deepfense.utils.registry import register_optimizer

try:
    from timm.optim import RAdam
except ImportError:
    RAdam = None


@register_optimizer("radam")
def RAdamOptimizer(params, config):
    """
    RAdam (Rectified Adam) optimizer.
    
    Requires: pip install timm
    """
    if RAdam is None:
        raise ImportError("RAdam requires 'timm' package. Install with: pip install timm")
    
    lr = config.get("lr", 0.001)
    betas = config.get("betas", (0.9, 0.999))
    weight_decay = config.get("weight_decay", 1e-4)
    eps = config.get("eps", 1e-8)
    
    return RAdam(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps
    )
```

## Example: Lookahead Optimizer Wrapper

Example of wrapping an optimizer with Lookahead:

```python
import torch
from deepfense.utils.registry import register_optimizer

try:
    from pytorch_optimizer import Lookahead
except ImportError:
    Lookahead = None


@register_optimizer("lookahead_adam")
def LookaheadAdamOptimizer(params, config):
    """
    Adam optimizer wrapped with Lookahead.
    
    Requires: pip install pytorch-optimizer
    """
    if Lookahead is None:
        raise ImportError("Lookahead requires 'pytorch-optimizer' package")
    
    # Base optimizer config
    base_lr = config.get("lr", 0.001)
    betas = config.get("betas", (0.9, 0.999))
    weight_decay = config.get("weight_decay", 1e-4)
    
    # Create base optimizer
    base_optimizer = torch.optim.Adam(
        params,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay
    )
    
    # Wrap with Lookahead
    lookahead_k = config.get("lookahead_k", 5)
    lookahead_alpha = config.get("lookahead_alpha", 0.5)
    
    return Lookahead(
        base_optimizer,
        k=lookahead_k,
        alpha=lookahead_alpha
    )
```

## Key Points

1. **Use @register_optimizer decorator**: Register with a unique string name
2. **Function signature**: Must accept `(params, config)` where `params` is an iterable of model parameters
3. **Return optimizer**: Must return a PyTorch optimizer instance
4. **Config parameters**: Extract parameters from the config dictionary
5. **Default values**: Provide sensible defaults for optional parameters
6. **No import needed**: Optimizers are registered when the module is imported

## Function Signature

Your optimizer function should follow this pattern:

```python
@register_optimizer("optimizer_name")
def MyOptimizer(params, config):
    """
    Args:
        params: Iterable of model parameters (typically model.parameters())
        config: Dictionary with optimizer configuration
            - lr: Learning rate (typically required)
            - Other optimizer-specific parameters
    
    Returns:
        torch.optim.Optimizer: PyTorch optimizer instance
    """
    lr = config.get("lr", 0.001)  # Extract with defaults
    # ... other parameters
    
    return torch.optim.SomeOptimizer(params, lr=lr, ...)
```

## Testing Your Optimizer

Test your optimizer before using it in training:

```python
import torch
import torch.nn as nn
from deepfense.training.optimizers import utils  # Import to register
from deepfense.utils.registry import build_optimizer

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Create optimizer config
optimizer_config = {
    "type": "my_optimizer",
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4
}

# Build optimizer
optimizer = build_optimizer(optimizer_config["type"], model.parameters(), optimizer_config)

# Test optimizer step
dummy_input = torch.randn(2, 10)
dummy_target = torch.randn(2, 1)
output = model(dummy_input)
loss = nn.MSELoss()(output, dummy_target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Optimizer step completed successfully!")
print(f"Optimizer type: {type(optimizer)}")
```

## Available PyTorch Optimizers

DeepFense supports any PyTorch optimizer. Common ones include:

- `torch.optim.SGD` - Stochastic Gradient Descent
- `torch.optim.Adam` - Adam
- `torch.optim.AdamW` - Adam with decoupled weight decay
- `torch.optim.RMSprop` - RMSprop
- `torch.optim.Adagrad` - Adagrad
- `torch.optim.Adadelta` - Adadelta

You can also use optimizers from external libraries like `timm` or `pytorch-optimizer`.

## Next Steps

- See [Adding Schedulers](adding_schedulers.md) for learning rate schedulers
- See [Training Guide](training_with_cli.md) for how to use optimizers in training
- See [Configuration Reference](../05_configuration.md) for full config options
- See existing optimizers in `deepfense/training/optimizers/utils.py` for reference

