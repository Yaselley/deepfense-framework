import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfense.data.temporal_utils import downsample_mask_to_frames
from deepfense.utils.registry import (
    register_detector,
    build_frontend,
    build_backend,
    build_loss,
)

logger = logging.getLogger(__name__)


@register_detector("TemporalDetector")
class TemporalDetector(nn.Module):
    """
    SSL frontend + frame-level backend + framewise loss for partial-deepfake detection.

    Pools SSL features from ``frontend_hop`` to ``label_hop`` when they differ.
    Uses the dataloader audio ``mask`` for SSL attention and frame-level masking.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frontend = build_frontend(
            config["frontend"]["type"], config["frontend"].get("args", {})
        )
        self.backend = build_backend(
            config["backend"]["type"], config["backend"].get("args", {})
        )

        self.frontend_hop = int(
            config.get("frontend_hop")
            or getattr(self.frontend, "frontend_hop", 320)
        )
        self.label_hop = self._resolve_label_hop(config)

        if self.label_hop < self.frontend_hop:
            raise ValueError(
                f"label_hop ({self.label_hop}) is finer than the SSL frontend's "
                f"native hop ({self.frontend_hop}). Pick a label_hop that is a "
                f"positive integer multiple of the frontend hop, or use a "
                f"higher-resolution frontend."
            )
        if self.label_hop % self.frontend_hop != 0:
            raise ValueError(
                f"label_hop ({self.label_hop}) is not an integer multiple of "
                f"frontend_hop ({self.frontend_hop}); pool factor would be "
                f"{self.label_hop / self.frontend_hop:.3f}. Pick a label_hop "
                f"that divides evenly."
            )
        self.pool_factor = self.label_hop // self.frontend_hop

        self.pool_mode = str(config.get("pool_mode", "mean")).lower()
        if self.pool_mode not in ("mean", "avg", "average", "max"):
            raise ValueError(
                f"pool_mode must be one of 'mean'/'avg'/'average' or 'max'; got {self.pool_mode!r}"
            )
        if self.pool_mode in ("avg", "average"):
            self.pool_mode = "mean"

        if self.pool_factor > 1:
            logger.info(
                "TemporalDetector pooling SSL features by %d using %s "
                "(frontend_hop=%d -> label_hop=%d)",
                self.pool_factor,
                self.pool_mode,
                self.frontend_hop,
                self.label_hop,
            )

        losses_cfg = config.get("loss")
        if isinstance(losses_cfg, dict):
            losses_cfg = [losses_cfg]
        if not losses_cfg:
            logger.warning("No losses configured for TemporalDetector.")
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

    @staticmethod
    def _resolve_label_hop(config) -> int:
        if config.get("label_hop") is not None:
            return int(config["label_hop"])
        if config.get("label_hop_ms") is not None:
            sr = int(config.get("sampling_rate", 16000))
            return int(round(float(config["label_hop_ms"]) * sr / 1000.0))
        if config.get("subsample") is not None:
            return int(config["subsample"])
        return int(config.get("frontend_hop", 320))

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.pool_factor <= 1:
            return features
        b, t, d = features.shape
        n_full = t // self.pool_factor
        if n_full == 0:
            return features.new_zeros((b, 0, d))
        feat = features[:, : n_full * self.pool_factor, :]
        feat = feat.reshape(b, n_full, self.pool_factor, d)
        if self.pool_mode == "max":
            feat = feat.max(dim=2).values
        else:
            feat = feat.mean(dim=2)
        return feat

    def _frame_mask_from_audio_mask(
        self, audio_mask: Optional[torch.Tensor], target_frames: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if audio_mask is None:
            return None
        return downsample_mask_to_frames(
            audio_mask.to(device).float(), self.label_hop, target_frames=target_frames
        )

    def forward(self, x, mask=None):
        if mask is not None and mask.device != x.device:
            mask = mask.to(x.device, non_blocking=True)

        features = self.frontend(x, mask=mask)
        pooled = self._pool_features(features)

        frame_mask = self._frame_mask_from_audio_mask(mask, pooled.size(1), pooled.device)
        if frame_mask is not None:
            pooled = pooled * frame_mask.unsqueeze(-1)

        frame_embeddings = self.backend(pooled)
        if frame_mask is not None and frame_embeddings.dim() == 3:
            frame_embeddings = frame_embeddings * frame_mask.unsqueeze(-1)

        scores = None
        logits = None
        probs = None

        if len(self.losses) > 0:
            main = self.losses[self.main_loss_idx]
            if hasattr(main, "get_logits"):
                logits = main.get_logits(frame_embeddings)
            if hasattr(main, "get_score"):
                scores = main.get_score(frame_embeddings)
            elif logits is not None:
                scores = logits
            if scores is not None:
                if scores.ndim == 3 and scores.shape[-1] > 1:
                    probs = F.softmax(scores, dim=-1)
                elif scores.ndim == 3:
                    probs = torch.sigmoid(scores)

        return {
            "embeddings": frame_embeddings,
            "scores": scores,
            "logits": logits,
            "probs": probs,
            "frame_mask": frame_mask,
        }

    def compute_loss(self, outputs, frame_targets):
        frame_embeddings = outputs["embeddings"]
        total = 0.0
        main_logits = outputs.get("logits")
        for i, (loss_module, w) in enumerate(zip(self.losses, self.loss_weights)):
            if i == self.main_loss_idx and main_logits is not None:
                contrib = w * loss_module(frame_embeddings, frame_targets, logits=main_logits)
            else:
                contrib = w * loss_module(frame_embeddings, frame_targets)
            total = total + contrib
        return total
