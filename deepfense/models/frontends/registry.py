FRONTEND = {}


def register_frontend(name: str):
    def decorator(cls):
        FRONTEND[name] = cls
        return cls

    return decorator


def build_frontend(name: str, config: dict):
    if name not in FRONTEND:
        raise ValueError(f"Unknown frontend: {name}")
    return FRONTEND[name](config)
