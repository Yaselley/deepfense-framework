# deepfense/models/losses/__init__.py
# This file ensures that all loss modules are imported and thus registered.

from . import cross_entropy
from . import a_softmax
from . import am_softmax
from . import oc_softmax
