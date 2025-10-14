EVAL_REGISTRY = {}

def register_eval(name):
    """Register a eval class under a given name."""
    def decorator(cls):
        EVAL_REGISTRY[name] = cls
        return cls
    return decorator

def build_loss(config):
    """Build loss from config dictionary"""
    eval_type = config.pop("type")
    if eval_type not in EVAL_REGISTRY:
        raise ValueError(f"Unknown loss: {eval_type}")
    return EVAL_REGISTRY[eval_type](**config)
