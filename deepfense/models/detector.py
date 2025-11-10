import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from deepfense.models.registry import DETECTOR, register_module
from deepfense.models.backends.registry import build_backend
from deepfense.models.frontends.registry import build_frontend
from deepfense.models.loss_mappers.registry import build_loss, build_mapper, get_mapper_for_loss

logger = logging.getLogger(__name__)

@register_module("StandardDetector")
class ModularDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frontend = build_frontend(config['frontend']["type"], config["frontend"].get('args', {}))
        self.backend = build_backend(config['backend']["type"], config["backend"].get('args', {}))

        # Build losses and their corresponding mappers
        losses_cfg = config.get("loss")
        if isinstance(losses_cfg, dict):
            losses_cfg = [losses_cfg]

        self.losses = nn.ModuleList()
        self.loss_weights = []
        self.mappers = nn.ModuleList()

        for loss_cfg in losses_cfg:
            cfg_copy = loss_cfg.copy()
            loss_type = cfg_copy.pop("type")
            self.losses.append(build_loss({"type": loss_type, **cfg_copy}))
            self.loss_weights.append(loss_cfg.get("weight", 1.0))

            # Automatically select mapper per loss
            mapper_cfg = {**loss_cfg, "type": get_mapper_for_loss(loss_type)} if get_mapper_for_loss(loss_type) else None
            if mapper_cfg:
                self.mappers.append(build_mapper(mapper_cfg.copy()))
            else:
                self.mappers.append(None)

    def forward(self, x, mask=None):
        features = self.frontend(x)
        logits = self.backend(features)
        outputs = []

        # Forward pass through each loss-specific mapper
        for mapper in self.mappers:
            if mapper:
                out = mapper(logits)
            else:
                out = logits
            outputs.append(out)

        # If only one output, simplify structure
        if len(outputs) == 1:
            outputs = outputs[0]

        # Compute probabilities for each output
        probs = []
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, tuple):
                    probs.append(F.softmax(out[0], dim=-1))
                elif isinstance(out, torch.Tensor):
                    probs.append(F.softmax(out, dim=-1))
                else:
                    probs.append(None)
        else:
            if isinstance(outputs, tuple):
                probs = F.softmax(outputs[0], dim=-1)
            elif isinstance(outputs, torch.Tensor):
                probs = F.softmax(outputs, dim=-1)
            else:
                probs = None

        return {"cls": outputs, "probs": probs}

    def compute_loss(self, outputs, targets):
        """Compute total weighted loss for all losses and their mappers."""
        total_loss = 0.0

        # Ensure outputs is a list for multi-loss case
        all_outputs = outputs['cls'] if isinstance(outputs['cls'], list) else [outputs['cls']]

        for loss_fn, w, out in zip(self.losses, self.loss_weights, all_outputs):
            total_loss += w * loss_fn(out, targets)

        return total_loss

