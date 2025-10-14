LOSS_REGISTRY = {}

def register_loss(name):
    """Register a loss class under a given name."""
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def build_loss(config):
    """Build loss from config dictionary"""
    loss_type = config.pop("type")
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {loss_type}")
    return LOSS_REGISTRY[loss_type](**config)
