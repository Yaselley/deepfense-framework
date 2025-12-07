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

        # Build losses (Unified)
        losses_cfg = config.get("loss")
        if isinstance(losses_cfg, dict):
            losses_cfg = [losses_cfg]
        
        if not losses_cfg:
             logger.warning("No losses configured!")
             losses_cfg = []

        self.losses = nn.ModuleList()
        self.loss_weights = []
        self.main_loss_idx = 0 # Default to first loss for scores
        self.main_loss_type = None # Store main loss type

        for i, loss_cfg in enumerate(losses_cfg):
            cfg_copy = loss_cfg.copy()
            loss_type = cfg_copy.pop("type")
            if i == self.main_loss_idx:
                self.main_loss_type = loss_type
            
            # Pass embedding_dim from backend config if not in loss config?
            # Usually backend output dim is needed. 
            # We might need to infer it or pass it explicitly.
            # For now assuming it's in the config or the Loss handles it.
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
        # Frontend
        features = self.frontend(x, mask=mask)
        # Backend
        embeddings = self.backend(features)
        
        # Get scores from the main loss module (for inference/validation)
        scores = None
        logits = None
        probs = None
        
        if len(self.losses) > 0:
            main_loss_module = self.losses[self.main_loss_idx]
            
            # 1. Get full logits/cosines for loss caching
            if hasattr(main_loss_module, "get_logits"):
                logits = main_loss_module.get_logits(embeddings)
            
            # 2. Get metric scores (1D LLR) for evaluation
            if hasattr(main_loss_module, "get_score"):
                 scores = main_loss_module.get_score(embeddings)
            elif logits is not None:
                 # Fallback if get_score is alias or missing
                 scores = logits
                
            if scores is not None:
                # Only apply softmax if scores are appropriate (e.g. logits)
                # For OC-Softmax, scores are cosines, so softmax might not be needed or meaningful the same way,
                # but usually it returns [N, 1], so softmax would just be 1.0. 
                # We keep it for standard classification flows.
                if scores.ndim > 1 and scores.shape[-1] > 1:
                     probs = F.softmax(scores, dim=-1)
                else:
                     probs = torch.sigmoid(scores) # Or just raw scores? Keeping it consistent.
        
        return {"embeddings": embeddings, "scores": scores, "logits": logits, "probs": probs}

    def compute_loss(self, outputs, targets):
        """
        Compute total weighted loss.
        outputs: Dict returned by forward() containing 'embeddings'.
        targets: Tensor of labels.
        """
        embeddings = outputs["embeddings"]
        total_loss = 0.0
        
        # We might have pre-computed LOGITS for the main loss (not scores!)
        main_logits = outputs.get("logits")

        for i, (loss_module, w) in enumerate(zip(self.losses, self.loss_weights)):
            # Optimization: Pass pre-computed LOGITS to main loss if available.
            if i == self.main_loss_idx and main_logits is not None:
                loss_val = loss_module(embeddings, targets, logits=main_logits)
            else:
                loss_val = loss_module(embeddings, targets)
                
            total_loss += w * loss_val
            
        return total_loss

