BACKEND = {}

def register_backend(name: str):
    def decorator(cls):
        BACKEND[name] = cls
        return cls
    return decorator

def build_backend(name: str, config: dict):
    if name not in BACKEND:
        raise ValueError(f"Unknown backend: {name}")
    return BACKEND[name](config)
