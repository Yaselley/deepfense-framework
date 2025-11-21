import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from deepfense.models.registry import DETECTOR, register_module
from deepfense.models.backends.registry import build_backend
from deepfense.models.frontends.registry import build_frontend
from deepfense.models.loss_mappers.registry import (
    build_loss,
    build_mapper,
    get_mapper_for_loss,
)

logger = logging.getLogger(__name__)


@register_module("StandardDetector")
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

        # Build losses and their corresponding mappers
        losses_cfg = config.get("loss")
        if isinstance(losses_cfg, dict):
            losses_cfg = [losses_cfg]

        self.losses = nn.ModuleList()
        self.loss_weights = []
        self.mappers = nn.ModuleList()
        self.main_loss = None

        for loss_cfg in losses_cfg:
            cfg_copy = loss_cfg.copy()
            loss_type = cfg_copy.pop("type")
            self.losses.append(build_loss({"type": loss_type, **cfg_copy}))
            self.loss_weights.append(loss_cfg.get("weight", 1.0))

            if self.main_loss == None:
                self.main_loss = loss_type

            # Automatically select mapper per loss
            mapper_cfg = (
                {**loss_cfg, "type": get_mapper_for_loss(loss_type)}
                if get_mapper_for_loss(loss_type)
                else None
            )
            if mapper_cfg:
                self.mappers.append(build_mapper(mapper_cfg.copy()))
            else:
                self.mappers.append(None)

    def forward(self, x, mask):
        """
        Runs the forward pass.

        Returns a dictionary:
        - "cls": A list of all mapper outputs, one for each loss.
                 Used by compute_loss().
        - "scores": A [B, C] tensor of raw scores (logits/cos_x) from the *first* loss.
                    Used for evaluation.
        - "probs": A [B, C] tensor of probabilities from the *first* loss.
                   (Softmax applied to "scores").
        """
        features = self.frontend(x)
        backend_output = self.backend(features)

        loss_inputs = []

        # 1. Get outputs for ALL mappers (for loss calculation)
        for mapper in self.mappers:
            if mapper:
                # Mapper produces loss-specific outputs
                # e.g., (B, n_classes) for CrossEntropyMapper
                # or ( (B, C), (B, C) ) for AMSoftmaxMapper
                out = mapper(backend_output)
            else:
                # Pass backend output directly if no mapper
                out = backend_output
            loss_inputs.append(out)

        if not loss_inputs:
            # Handle case with no losses configured
            logger.warning(
                "Forward pass executed but no losses/mappers are configured."
            )
            return {"cls": [], "scores": None, "probs": None}

        # 2. Get scores and probabilities from the FIRST mapper's output
        primary_output = loss_inputs[0]
        scores_tensor = None
        probs_tensor = None

        try:
            if isinstance(primary_output, tuple):
                # Handle tuple outputs like AMSoftmax (cos_x, phi_x)
                # We use the first element (cos_x) for inference scores
                scores_tensor = primary_output[0]

            elif isinstance(primary_output, torch.Tensor):
                # Handle tensor outputs like CrossEntropy (logits)
                scores_tensor = primary_output

            # Now, calculate probs from the determined scores_tensor
            if isinstance(scores_tensor, torch.Tensor):
                probs_tensor = F.softmax(scores_tensor, dim=-1)
            else:
                logger.warning(
                    f"Could not determine scores_tensor from primary output type: {type(primary_output)}"
                )

        except Exception as e:
            logger.error(f"Error calculating scores/probs: {e}")
            if isinstance(primary_output, tuple):
                logger.error(
                    f"Primary output was a tuple. Shapes: {[item.shape for item in primary_output if isinstance(item, torch.Tensor)]}"
                )
            elif isinstance(primary_output, torch.Tensor):
                logger.error(
                    f"Primary output was a tensor. Shape: {primary_output.shape}"
                )

        return {"cls": loss_inputs, "scores": scores_tensor, "probs": probs_tensor}

    def compute_loss(self, outputs, targets):
        """Compute total weighted loss for all losses and their mappers."""
        total_loss = 0.0

        # 'cls' is now guaranteed to be a list from the forward pass
        all_loss_inputs = outputs["cls"]

        if not all_loss_inputs:
            logger.warning(
                "compute_loss called, but no loss inputs found in outputs['cls']."
            )
            return 0.0

        if len(all_loss_inputs) != len(self.losses):
            logger.error(
                f"Mismatch in compute_loss: {len(all_loss_inputs)} outputs but {len(self.losses)} losses."
            )
            return 0.0

        for loss_fn, w, loss_input in zip(
            self.losses, self.loss_weights, all_loss_inputs
        ):
            if loss_input is None:
                logger.warning(
                    f"Got None input for loss {loss_fn.__class__.__name__}, skipping."
                )
                continue

            try:
                total_loss += w * loss_fn(loss_input, targets)
            except Exception as e:
                logger.error(f"Error computing loss {loss_fn.__class__.__name__}: {e}")
                logger.error(f"Input type: {type(loss_input)}")
                if isinstance(loss_input, tuple):
                    logger.error(
                        f"Input tuple shapes: {[i.shape for i in loss_input if isinstance(i, torch.Tensor)]}"
                    )
                elif isinstance(loss_input, torch.Tensor):
                    logger.error(f"Input tensor shape: {loss_input.shape}")
                logger.error(f"Target shape: {targets.shape}")
                raise e  # Re-raise the exception after logging

        return total_loss
