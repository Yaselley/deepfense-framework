from .registry import register_transform

# Import all transform modules
from . import transforms
from . import augmentations

# Register all functions in transforms.py
for name in dir(transforms):
    fn = getattr(transforms, name)
    if callable(fn) and not name.startswith("_"):
        register_transform(name)(fn)

# Register all functions in augmentations.py
for name in dir(augmentations):
    fn = getattr(augmentations, name)
    if callable(fn) and not name.startswith("_"):
        register_transform(name)(fn)
