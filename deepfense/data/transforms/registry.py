TRANSFORM_REGISTRY = {}


def register_transform(name):
    """
    Decorator to register a transform function or class under a given name.

    Example:
        @register_transform("normalize")
        def normalize(x, mean=0.0, std=1.0):
            ...
    """

    def decorator(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return decorator


def get_transform(name):
    """
    Retrieve a transform from the registry by name.
    """
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Transform '{name}' is not registered.")
    return TRANSFORM_REGISTRY[name]


def build_transform(config):
    """
    Build a transform from a config dictionary.

    Example config:
        {"type": "normalize", "mean": 0.0, "std": 1.0}
    """
    config = config.copy()  # avoid modifying original
    transform_type = config.pop("type")
    transform_cls_or_fn = get_transform(transform_type)

    # If it's a class, instantiate it with the config parameters
    if isinstance(transform_cls_or_fn, type):
        return transform_cls_or_fn(**config)
    else:
        # If it's a function, return a lambda that calls it with the config parameters
        return lambda x: transform_cls_or_fn(x, **config)


def build_transforms_from_config(config_list):
    """
    Build a pipeline of transforms from a list of configs.
    Returns a function that applies them sequentially.

    Example:
        configs = [
            {"type": "normalize", "mean": 0.0, "std": 1.0},
            {"type": "add_noise", "noise_level": 0.01}
        ]
        transform_pipeline = build_transforms_from_config(configs)
        x = transform_pipeline(x)
    """
    if not config_list:
        # Return an identity transform (no-op)
        return None

    transforms = [build_transform(cfg) for cfg in config_list]

    def pipeline(x):
        for t in transforms:
            x = t(x)
        return x

    return pipeline
