# Extending DeepFense

DeepFense is designed to be easily extensible.

## Adding a New Model

1.  **Create the file**: `deepfense/models/backends/my_model.py`.
2.  **Implement the class**: Inherit from `nn.Module`.
3.  **Register it**:

    ```python
    from deepfense.utils.registry import register_backend
    import torch.nn as nn

    @register_backend("MyModel")
    class MyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layer = nn.Linear(config["input_dim"], config["output_dim"])

        def forward(self, x):
            return self.layer(x)
    ```

4.  **Import it**: Add `from . import my_model` in `deepfense/models/backends/__init__.py`.
5.  **Use it**: Update `train.yaml`.

    ```yaml
    backend:
      type: "MyModel"
      args:
        input_dim: 1024
        output_dim: 32
    ```

## Adding a New Loss

Follow the same pattern with `@register_loss` in `deepfense/models/losses/`.

## Adding a New Augmentation

Follow the same pattern with `@register_transform` in `deepfense/data/transforms/augmentations.py`.
