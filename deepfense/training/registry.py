# training/registry.py
TRAINER_REGISTRY = {}
TRAINER_CONFIG_REGISTRY = {}

def register_trainer(name):
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator

def get_trainer(name):
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name]


def register_config_trainer(name):
    def decorator(cls):
        TRAINER_CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def get_trainer_config(name):
    if name not in TRAINER_CONFIG_REGISTRY:
        raise ValueError(f"Trainer Config'{name}' not found. Available: {list(TRAINER_CONFIG_REGISTRY.keys())}")
    return TRAINER_CONFIG_REGISTRY[name]