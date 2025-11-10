SCHEDULER_REGISTRY = {}

def register_scheduler(name):
    """Register a scheduler function under a given name."""
    def decorator(fn):
        SCHEDULER_REGISTRY[name] = fn
        return fn
    return decorator

def build_scheduler(config):
    scheduler_type = config.pop("type")
    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    return SCHEDULER_REGISTRY[scheduler_type](config)
