import torch
import torch.nn as nn
import logging
from deepfense.models.registry import DETECTOR, register_module
from deepfense.models.backends.registry import build_backend
from deepfense.models.frontends.registry import build_frontend

logger = logging.getLogger(__name__)

@register_module("StandardDetector")
class ModularDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # build frontend and backend
        self.frontend = build_frontend(config['frontend']["type"], config["frontend"].get('args', {}))
        self.backend = build_backend(config['backend']["type"], config["backend"].get('args', {}))

    def forward(self, x, mask=None):
        features = self.frontend(x)
        logits = self.backend(features)
        return {"cls": logits}
