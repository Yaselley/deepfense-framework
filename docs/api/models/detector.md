# StandardDetector

`deepfense.models.detector.ModularDetector`

The main container class that connects the pipeline.

## Class Definition

```python
class ModularDetector(nn.Module):
    def __init__(self, config):
        # Builds frontend, backend, and losses from config
        ...

    def forward(self, x, mask=None):
        """
        Returns dict:
        - "embeddings": Backend output
        - "scores": Main loss scores (for validation)
        - "probs": Softmax probabilities
        - "logits": Raw logits
        """
        ...

    def compute_loss(self, outputs, targets):
        """
        Computes weighted sum of all configured losses.
        """
        ...
```

## Configuration

```yaml
model:
  type: "StandardDetector"
  frontend: { ... }
  backend: { ... }
  loss:
    - { type: "Loss1", weight: 0.5, ... }
    - { type: "Loss2", weight: 0.5, ... }
```

