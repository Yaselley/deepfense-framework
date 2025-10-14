OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    """Register a optimizer class under a given name."""
    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

def build_optimizer(config):
    """Build loss from config dictionary"""
    optimizer_type = config.pop("type")
    if optimizer_type not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown loss: {optimizer_type}")
    return OPTIMIZER_REGISTRY[optimizer_type](**config)
