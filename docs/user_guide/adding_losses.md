# Adding a New Loss Function

This guide shows you how to add a custom loss function to DeepFense.

## Overview

Loss functions in DeepFense combine a mapping layer (classifier/projection) with the actual loss computation. They must inherit from `BaseLoss` and be registered with `@register_loss`. Loss functions compute classification loss and provide scores for inference.

## Step-by-Step Guide

### Step 1: Create the Loss File

Create a new file `deepfense/models/losses/my_loss.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


@register_loss("MyContrastiveLoss")
class MyContrastiveLoss(BaseLoss):
    """
    Custom contrastive loss for audio spoofing detection.
    
    Args:
        config: Dictionary containing configuration parameters
            - embedding_dim: Embedding dimension (from backend output)
            - n_classes: Number of classes (default: 2)
            - temperature: Temperature parameter for contrastive loss
            - margin: Margin for contrastive learning
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = config["embedding_dim"]  # From backend output
        n_classes = config.get("n_classes", 2)
        self.temperature = config.get("temperature", 0.07)
        self.margin = config.get("margin", 1.0)
        
        # Classification head (mapper)
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, targets, logits=None):
        """
        Compute loss.
        
        Args:
            embeddings: [Batch, embedding_dim] - embeddings from backend
            targets: [Batch] - target labels (0 or 1)
            logits: Optional pre-computed logits to avoid recalculation
            
        Returns:
            Loss scalar tensor
        """
        # Compute logits if not provided
        if logits is None:
            logits = self.classifier(embeddings)
        
        # Compute classification loss
        loss = self.ce_loss(logits, targets.long())
        
        return loss
    
    def get_score(self, embeddings):
        """
        Get scores for inference/evaluation.
        
        Args:
            embeddings: [Batch, embedding_dim]
            
        Returns:
            Scores [Batch] - bonafide scores (higher = more bonafide)
        """
        logits = self.classifier(embeddings)
        
        # For binary classification, return bonafide score - spoof score
        if logits.shape[1] == 2:
            # Return logit difference: bonafide - spoof
            return logits[:, 1] - logits[:, 0]
        else:
            # For multi-class, return the bonafide class logit
            return logits[:, self.bonafide_label]
    
    def get_logits(self, embeddings):
        """
        Get raw logits for caching/loss computation.
        
        Args:
            embeddings: [Batch, embedding_dim]
            
        Returns:
            Logits [Batch, n_classes]
        """
        return self.classifier(embeddings)
```

### Step 2: Register in __init__.py

Import your loss in `deepfense/models/losses/__init__.py`:

```python
from .cross_entropy import CrossEntropy
from .oc_softmax import OCSoftmaxLoss
from .am_softmax import AMSoftmaxLoss
from .a_softmax import ASoftmaxLoss
from .my_loss import MyContrastiveLoss  # Add this line
```

**Important**: The import statement is required for the decorator to register the loss when the module is loaded.

### Step 3: Use in Configuration

Use your loss in a YAML configuration file:

```yaml
model:
  type: "Detector"
  loss:
    - type: "MyContrastiveLoss"  # Your registered name
      weight: 1.0
      embedding_dim: 128  # Must match backend output_dim
      n_classes: 2
      temperature: 0.1
      margin: 1.0
```

**Note**: Losses are specified as a list, so you can use multiple losses with different weights.

### Step 4: Verify Registration

Check that your loss is registered:

```bash
deepfense list --component-type losses
```

Or programmatically:

```python
from deepfense.models.losses import *  # Import all losses
from deepfense.utils.registry import LOSS_REGISTRY

# Check if registered
if "MyContrastiveLoss" in LOSS_REGISTRY:
    print("Loss registered successfully!")
    print("Available losses:", LOSS_REGISTRY.list())
```

## Complete Example: Focal Loss

Here's a complete example of a Focal Loss implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


@register_loss("FocalLoss")
class FocalLoss(BaseLoss):
    """
    Focal Loss for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = config["embedding_dim"]
        n_classes = config.get("n_classes", 2)
        self.alpha = config.get("alpha", 0.25)  # Weighting factor
        self.gamma = config.get("gamma", 2.0)   # Focusing parameter
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
    
    def forward(self, embeddings, targets, logits=None):
        """
        Compute focal loss.
        """
        if logits is None:
            logits = self.classifier(embeddings)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Compute p_t
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
    def get_score(self, embeddings):
        """Get scores for inference."""
        logits = self.classifier(embeddings)
        if logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        return logits[:, self.bonafide_label]
    
    def get_logits(self, embeddings):
        """Get raw logits."""
        return self.classifier(embeddings)
```

## Example: Angular Margin Loss (AM-Softmax style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfense.models.base_model import BaseLoss
from deepfense.utils.registry import register_loss


@register_loss("AngularMarginLoss")
class AngularMarginLoss(BaseLoss):
    """
    Angular margin loss similar to AM-Softmax.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = config["embedding_dim"]
        n_classes = config.get("n_classes", 2)
        self.margin = config.get("margin", 0.3)
        self.scale = config.get("scale", 30.0)
        
        # Weight matrix for cosine similarity
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, targets, logits=None):
        """
        Compute angular margin loss.
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)
        
        # Add margin
        target_one_hot = torch.zeros_like(cosine)
        target_one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        
        # Apply margin only to positive class
        output = (1 - target_one_hot) * cosine + target_one_hot * (cosine - self.margin)
        output *= self.scale
        
        # Compute cross entropy
        return F.cross_entropy(output, targets.long())
    
    def get_score(self, embeddings):
        """Get cosine similarity scores."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)
        
        if cosine.shape[1] == 2:
            return cosine[:, 1] - cosine[:, 0]
        return cosine[:, self.bonafide_label]
    
    def get_logits(self, embeddings):
        """Get cosine similarity logits."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings_norm, weight_norm)
```

## Key Points

1. **Inherit from BaseLoss**: Provides `bonafide_label` and `spoof_label` attributes
2. **Use @register_loss decorator**: Register with a unique string name
3. **Implement forward()**: Must accept embeddings and targets, return loss scalar
4. **Implement get_score()**: Returns scores for inference (higher = more bonafide)
5. **Implement get_logits()**: Returns raw logits/similarities for caching
6. **Import in __init__.py**: Critical for the decorator to execute
7. **Match embedding_dim**: Must match the backend's `output_dim`

## Required Methods

Your loss class must implement:

- **`forward(embeddings, targets, logits=None)`**: Compute and return loss scalar
- **`get_score(embeddings)`**: Return scores for inference (shape: [Batch])
- **`get_logits(embeddings)`**: Return raw logits (shape: [Batch, n_classes])

## Testing Your Loss

Test your loss before using it in training:

```python
import torch
from deepfense.models.losses.my_loss import MyContrastiveLoss

# Create loss instance
config = {
    "embedding_dim": 128,
    "n_classes": 2,
    "temperature": 0.1
}
loss_fn = MyContrastiveLoss(config)

# Test forward pass
dummy_embeddings = torch.randn(4, 128)  # [Batch=4, embedding_dim=128]
dummy_targets = torch.tensor([1, 0, 1, 0])  # [Batch=4]

loss_value = loss_fn(dummy_embeddings, dummy_targets)
print(f"Loss value: {loss_value.item()}")

# Test get_score
scores = loss_fn.get_score(dummy_embeddings)
print(f"Scores shape: {scores.shape}")  # Should be [4]

# Test get_logits
logits = loss_fn.get_logits(dummy_embeddings)
print(f"Logits shape: {logits.shape}")  # Should be [4, 2]
```

## Using Multiple Losses

You can combine multiple losses in your config:

```yaml
model:
  loss:
    - type: "CrossEntropy"
      weight: 0.5
      embedding_dim: 128
      n_classes: 2
    - type: "MyContrastiveLoss"
      weight: 0.5
      embedding_dim: 128
      n_classes: 2
      temperature: 0.1
```

The total loss will be: `weight1 * loss1 + weight2 * loss2`.

## Next Steps

- See [Adding a New Backend](adding_backends.md) for backend creation
- See [Training Guide](training_with_cli.md) for how to train with your custom loss
- See [Configuration Reference](../05_configuration.md) for full config options

