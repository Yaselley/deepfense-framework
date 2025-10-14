import torch
import torch.nn as nn
import logging
from deepfense.models.registry import DETECTOR, register_module
from deepfense.models.backends.registry import build_backend
from deepfense.models.frontends.registry import build_frontend

logger = logging.getLogger(__name__)

@register_module("detector_modular")
class ModularDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # build frontend and backend
        self.frontend = build_frontend(config['frontend']["name"], config["frontend"].get('args', {}))
        self.backend = build_backend(config['backend']["name"], config["backend"].get('args', {}))
        
        self.outdim = self.backend.get_outdim
        self.projection = nn.Linear(self.outdim,config['projection']["n_classes"])

    def forward(self, x):
        features = self.frontend.forward(x)
        features = self.backend(features)
        logits = self.projection(features)
        return logits
