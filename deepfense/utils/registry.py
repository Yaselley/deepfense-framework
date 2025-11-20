# deepfense/utils/registry.py


class Registry:
    """Simple registry to map string keys to classes or callables."""

    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name):
        def decorator(obj):
            if name in self._registry:
                raise KeyError(f"{name} already registered in {self._name}")
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name]

    def build(self, name, **kwargs):
        cls = self.get(name)
        return cls(**kwargs)

    def list(self):
        return list(self._registry.keys())


# Instantiate global registries
MODEL_REGISTRY = Registry("Model")
DATASET_REGISTRY = Registry("Dataset")
LOSS_REGISTRY = Registry("Loss")
METRIC_REGISTRY = Registry("Metric")
AUGMENTATION_REGISTRY = Registry("Augmentation")
INPUT_REGSITRY = Registry("Input")
