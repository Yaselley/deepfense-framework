TRANSFORM_REGISTRY = {}

def register_transform(name):
    """Register a transform function under a given name."""
    def decorator(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn
    return decorator

def build_transform(config):
    trasnform_type = config.pop("type")
    if trasnform_type not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown scheduler: {trasnform_type}")
    return TRANSFORM_REGISTRY[trasnform_type](**config)
