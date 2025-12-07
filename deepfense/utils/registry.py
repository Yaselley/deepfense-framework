import logging

logger = logging.getLogger(__name__)

class Registry:
    """Simple registry to map string keys to classes or callables."""

    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name):
        def decorator(obj):
            if name in self._registry:
                logger.warning(f"{name} already registered in {self._name}. Overwriting.")
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self._name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def build(self, name, *args, **kwargs):
        obj = self.get(name)
        return obj(*args, **kwargs)

    def list(self):
        return list(self._registry.keys())

    def __contains__(self, key):
        return key in self._registry


# Instantiate global registries
# Models
DETECTOR_REGISTRY = Registry("Detector")
FRONTEND_REGISTRY = Registry("Frontend")
BACKEND_REGISTRY = Registry("Backend")

# Losses (Unified)
LOSS_REGISTRY = Registry("Loss")

# Data
DATASET_REGISTRY = Registry("Dataset")
TRANSFORM_REGISTRY = Registry("Transform")

# Training
TRAINER_REGISTRY = Registry("Trainer")
OPTIMIZER_REGISTRY = Registry("Optimizer")
SCHEDULER_REGISTRY = Registry("Scheduler")

# Metrics
METRIC_REGISTRY = Registry("Metric")


# -----------------------------------------------------------------------------
# Helper functions for registration and building
# -----------------------------------------------------------------------------

# --- Detector ---
def register_detector(name):
    return DETECTOR_REGISTRY.register(name)

def build_detector(name, config):
    return DETECTOR_REGISTRY.build(name, config)

# --- Frontend ---
def register_frontend(name):
    return FRONTEND_REGISTRY.register(name)

def build_frontend(name, config):
    return FRONTEND_REGISTRY.build(name, config)

# --- Backend ---
def register_backend(name):
    return BACKEND_REGISTRY.register(name)

def build_backend(name, config):
    return BACKEND_REGISTRY.build(name, config)

# --- Loss ---
def register_loss(name):
    return LOSS_REGISTRY.register(name)

def build_loss(name, config):
    return LOSS_REGISTRY.build(name, config)

# --- Dataset ---
def register_dataset(name):
    return DATASET_REGISTRY.register(name)

def build_dataset(name, **kwargs):
    return DATASET_REGISTRY.build(name, **kwargs)

# --- Transform ---
def register_transform(name):
    return TRANSFORM_REGISTRY.register(name)

def build_transform(name, **kwargs):
    return TRANSFORM_REGISTRY.build(name, **kwargs)

def build_transforms_pipeline(config_list):
    """
    Build a pipeline of transforms from a list of configs.
    Returns a function that applies them sequentially.
    """
    if not config_list:
        return None

    # Pre-build partial functions or callables
    transforms = []
    for cfg in config_list:
        cfg_copy = cfg.copy()
        t_type = cfg_copy.pop("type")
        t_obj = TRANSFORM_REGISTRY.get(t_type)
        
        if isinstance(t_obj, type):
             try:
                 transforms.append(t_obj(**cfg_copy))
             except Exception as e:
                 raise ValueError(f"Failed to instantiate transform {t_type}: {e}")
        elif callable(t_obj):
            def make_transform(func, kwargs):
                return lambda x: func(x, **kwargs)
            transforms.append(make_transform(t_obj, cfg_copy))
        else:
             raise ValueError(f"Transform {t_type} is neither a class nor a callable function.")

    def pipeline(x):
        for t in transforms:
            x = t(x)
        return x

    return pipeline

# --- Trainer ---
def register_trainer(name):
    return TRAINER_REGISTRY.register(name)

def build_trainer(name, **kwargs):
    return TRAINER_REGISTRY.build(name, **kwargs)

# --- Optimizer ---
def register_optimizer(name):
    return OPTIMIZER_REGISTRY.register(name)

def build_optimizer(name, params, config):
    # Optimizers usually need params as first arg
    return OPTIMIZER_REGISTRY.build(name, params, config)

def get_optimizer(name):
    return OPTIMIZER_REGISTRY.get(name)

# --- Scheduler ---
def register_scheduler(name):
    return SCHEDULER_REGISTRY.register(name)

def build_scheduler(name, optimizer, config):
    # Schedulers usually need optimizer as first arg
    return SCHEDULER_REGISTRY.build(name, optimizer, config)

def get_scheduler(name):
    return SCHEDULER_REGISTRY.get(name)

# --- Metric ---
def register_metric(name):
    return METRIC_REGISTRY.register(name)

def build_metric(name, **kwargs):
    return METRIC_REGISTRY.build(name, **kwargs)
