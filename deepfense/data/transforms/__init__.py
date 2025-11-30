from deepfense.utils.registry import register_transform

# Import all transform modules
from . import transforms
from . import augmentations

# The decorators in the imported modules handle registration.
# No need to manually iterate and register again.
