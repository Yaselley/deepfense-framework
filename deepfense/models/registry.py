DETECTOR = {}

def register_module(module_name: str):
    """Decorator to register detector classes."""
    def decorator(cls):
        DETECTOR[module_name] = cls
        return cls
    return decorator

def build_detector(config: dict):
    """Build detector from config dictionary."""
    detector_type = config.get("detector")
    if detector_type not in DETECTOR:
        raise ValueError(f"Unknown detector: {detector_type}")
    return DETECTOR[detector_type](config)
