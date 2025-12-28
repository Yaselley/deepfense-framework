# Adding a New Backend

This guide shows you how to add a custom backend to DeepFense using the library's registry system.

## Overview

Backends in DeepFense are responsible for processing frontend features and producing classification embeddings. They must inherit from `BaseBackend` and be registered with `@register_backend`.

## Step-by-Step Guide

### Step 1: Create the Backend File

Create a new file `deepfense/models/backends/my_backend.py`:

```python
import torch
import torch.nn as nn
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend


@register_backend("MyBackend")
class MyBackend(BaseBackend):
    """
    Custom backend for audio spoofing detection.
    
    Args:
        config: Dictionary containing configuration parameters
            - input_dim: Input dimension (from frontend output)
            - output_dim: Output embedding dimension
            - hidden_dim: Hidden layer dimension (optional)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Access input_dim from BaseBackend
        input_dim = self.input_dim
        output_dim = config.get("output_dim", 128)
        hidden_dim = config.get("hidden_dim", 256)
        
        # Build your network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._output_dim = output_dim
    
    def forward(self, x, **kwargs):
        """
        Forward pass.
        
        Args:
            x: Features [Batch, Time, Dim] or [Batch, Dim]
            **kwargs: Additional arguments
            
        Returns:
            Embeddings [Batch, output_dim]
        """
        # Handle different input shapes
        if x.dim() == 3:
            # [Batch, Time, Dim] - pool over time dimension
            x = x.mean(dim=1)  # Average pooling
        # else: already [Batch, Dim]
        
        return self.network(x)
    
    @property
    def output_dim(self):
        """Return output dimension."""
        return self._output_dim
```

### Step 2: Register in __init__.py

Import your backend in `deepfense/models/backends/__init__.py`:

```python
from .mlp import MLP
from .aasist import AASIST
from .nes2net import Nes2Net
from .tcm import TCM_Conformer
from .ecapa_tdnn import ECAPA_TDNN
from .rawnet import RawNet2
from .my_backend import MyBackend  # Add this line
```

**Important**: The import statement is required to register the backend with the decorator when the module is loaded.

### Step 3: Use in Configuration

Use your backend in a YAML configuration file:

```yaml
model:
  type: "Detector"
  backend:
    type: "MyBackend"  # Your registered name
    args:
      input_dim: 768   # Must match frontend output dimension
      output_dim: 128
      hidden_dim: 256
```

### Step 4: Verify Registration

Check that your backend is registered:

```bash
deepfense list --component-type backends
```

Or programmatically:

```python
from deepfense.models.backends import *  # Import all backends
from deepfense.utils.registry import BACKEND_REGISTRY

# Check if registered
if "MyBackend" in BACKEND_REGISTRY:
    print("Backend registered successfully!")
    print("Available backends:", BACKEND_REGISTRY.list())
```

## Complete Example

Here's a complete example of a more complex backend with attention pooling:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.utils.registry import register_backend
from deepfense.models.base_model import BaseBackend


@register_backend("AttentionBackend")
class AttentionBackend(BaseBackend):
    """
    Backend with attention-based pooling.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        input_dim = self.input_dim
        output_dim = config.get("output_dim", 128)
        hidden_dim = config.get("hidden_dim", 256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Projection layer
        self.projection = nn.Linear(input_dim, output_dim)
        
        self._output_dim = output_dim
    
    def forward(self, x, **kwargs):
        """
        Args:
            x: [Batch, Time, Dim]
        Returns:
            [Batch, output_dim]
        """
        # Compute attention weights
        att_weights = self.attention(x)  # [Batch, Time, 1]
        att_weights = F.softmax(att_weights, dim=1)
        
        # Weighted sum
        x = torch.sum(x * att_weights, dim=1)  # [Batch, Dim]
        
        # Project to output dimension
        return self.projection(x)
    
    @property
    def output_dim(self):
        return self._output_dim
```

## Key Points

1. **Inherit from BaseBackend**: Provides `input_dim` from the frontend automatically
2. **Use @register_backend decorator**: Register with a unique string name
3. **Implement forward()**: Must return embeddings of shape [Batch, output_dim]
4. **Set output_dim property**: Required for loss functions to know embedding size
5. **Import in __init__.py**: Critical for the decorator to execute during module import
6. **Handle variable input shapes**: Frontends may return [Batch, Time, Dim] or [Batch, Dim]

## Testing Your Backend

Test your backend before using it in training:

```python
import torch
from deepfense.models.backends.my_backend import MyBackend

# Create backend instance
config = {
    "input_dim": 768,
    "output_dim": 128,
    "hidden_dim": 256
}
backend = MyBackend(config)

# Test forward pass
dummy_input = torch.randn(4, 100, 768)  # [Batch=4, Time=100, Dim=768]
output = backend(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Should be [4, 128]

# Test with already pooled input
dummy_input_pooled = torch.randn(4, 768)  # [Batch=4, Dim=768]
output_pooled = backend(dummy_input_pooled)
print(f"Pooled input shape: {dummy_input_pooled.shape}")
print(f"Output shape: {output_pooled.shape}")  # Should be [4, 128]
```

## Next Steps

- See [Adding a New Frontend](adding_frontends.md) for frontend creation
- See [Adding a New Loss](extending.md#adding-a-new-loss) for loss functions
- See [Training Guide](training.md) for how to train with your custom backend

