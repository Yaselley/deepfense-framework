DATASET_REGISTRY = {}


def register_dataset(name):
    """
    Decorator to register a dataset class by name.
    Example:
        @register_dataset("detection")
        class DetectionDataset(BaseDataset):
            ...
    """
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_class(name):
    """
    Retrieves a dataset class from the registry by name.
    Raises a clear error if the dataset is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered.")
    return DATASET_REGISTRY[name]
