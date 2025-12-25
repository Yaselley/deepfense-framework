import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from deepfense.utils.registry import (
    register_detector,
    build_frontend,
    build_backend,
    build_loss,
)

logger = logging.getLogger(__name__)


@register_detector("StandardDetector")
class ModularDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frontend = build_frontend(
            config["frontend"]["type"], config["frontend"].get("args", {})
        )
        self.backend = build_backend(
            config["backend"]["type"], config["backend"].get("args", {})
        )

        losses_cfg = config.get("loss")
        if isinstance(losses_cfg, dict):
            losses_cfg = [losses_cfg]
        
        if not losses_cfg:
             logger.warning("No losses configured!")
             losses_cfg = []

        self.losses = nn.ModuleList()
        self.loss_weights = []
        self.main_loss_idx = 0
        self.main_loss_type = None

        for i, loss_cfg in enumerate(losses_cfg):
            cfg_copy = loss_cfg.copy()
            loss_type = cfg_copy.pop("type")
            if i == self.main_loss_idx:
                self.main_loss_type = loss_type
            
            self.losses.append(build_loss(loss_type, cfg_copy))
            self.loss_weights.append(loss_cfg.get("weight", 1.0))

    def forward(self, x, mask=None):
        """
        Runs the forward pass.
        Returns a dictionary with:
        - "embeddings": The output of the backend.
        - "scores": Tensor of scores for validation (from the main loss).
        - "probs": Tensor of probabilities (softmax of scores).
        """
        features = self.frontend(x, mask=mask)
        embeddings = self.backend(features)
        
        scores = None
        logits = None
        probs = None
        
        if len(self.losses) > 0:
            main_loss_module = self.losses[self.main_loss_idx]
            
            if hasattr(main_loss_module, "get_logits"):
                logits = main_loss_module.get_logits(embeddings)
            
            if hasattr(main_loss_module, "get_score"):
                 scores = main_loss_module.get_score(embeddings)
            elif logits is not None:
                 # Fallback if get_score is alias or missing
                 scores = logits
                
            if scores is not None:
                if scores.ndim > 1 and scores.shape[-1] > 1:
                     probs = F.softmax(scores, dim=-1)
                else:
                     probs = torch.sigmoid(scores)
        
        return {"embeddings": embeddings, "scores": scores, "logits": logits, "probs": probs}

    def compute_loss(self, outputs, targets):
        """
        Compute total weighted loss.
        outputs: Dict returned by forward() containing 'embeddings'.
        targets: Tensor of labels.
        """
        embeddings = outputs["embeddings"]
        total_loss = 0.0

        main_logits = outputs.get("logits")

        for i, (loss_module, w) in enumerate(zip(self.losses, self.loss_weights)):
            if i == self.main_loss_idx and main_logits is not None:
                loss_val = loss_module(embeddings, targets, logits=main_logits)
            else:
                loss_val = loss_module(embeddings, targets)
                
            total_loss += w * loss_val
            
        return total_loss

