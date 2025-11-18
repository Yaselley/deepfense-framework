from typing import Dict, Type
from torch import nn

LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}
MAPPER_REGISTRY: Dict[str, Type[nn.Module]] = {}

LOSS_TO_MAPPER = {
    "OCSoftmax": "OCSoftmaxMapper",
    "ASoftmax": "ASoftmaxMapper",
    "AMSoftmax": "AMSoftmaxMapper",
    "CrossEntropy": "CrossEntropyMapper",  # no special mapper needed
}

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
    return LOSS_REGISTRY[loss_type](config)

def register_mapper(name):
    """Register a mapper class under a given name."""
    def decorator(cls):
        MAPPER_REGISTRY[name] = cls
        return cls
    return decorator

def build_mapper(config):
    """Build mapper from config dictionary"""
    mapper_type = config.pop("type")
    if mapper_type not in MAPPER_REGISTRY:
        raise ValueError(f"Unknown mapper: {mapper_type}")
    return MAPPER_REGISTRY[mapper_type](config)

def get_mapper_for_loss(loss_type: str):
    """Return the default mapper name associated with a given loss type."""
    return LOSS_TO_MAPPER.get(loss_type, None)