# Registry System

`deepfense.utils.registry.Registry`

The core mechanism for decoupling configuration from implementation.

## Global Registries

*   `DETECTOR_REGISTRY`
*   `FRONTEND_REGISTRY`
*   `BACKEND_REGISTRY`
*   `LOSS_REGISTRY`
*   `DATASET_REGISTRY`
*   `TRANSFORM_REGISTRY`
*   `TRAINER_REGISTRY`

## Usage

**Registering a new class**:
```python
from deepfense.utils.registry import register_backend

@register_backend("MyNewNet")
class MyNetwork(nn.Module):
    def __init__(self, config):
        ...
```

**Using it in Config**:
```yaml
backend:
  type: "MyNewNet"
  args: ...
```

