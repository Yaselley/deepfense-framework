"""Trainer-time waveform augmentation (Adapters/main.py style)."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch

from deepfense.utils.registry import build_transform

logger = logging.getLogger(__name__)


def build_batch_augmenter(config: dict[str, Any] | None) -> BatchAugmenter | None:
    if not config or not config.get("enabled", False):
        return None
    return BatchAugmenter(config)


class BatchAugmenter:
    """
    Apply augmentations inside the trainer on the batch waveform tensor.

    Mirrors ``Adapters/main.py``:
      - one random augmentation per sample (``mode: parallel``)
      - optional ``concat_original: true`` → ``cat([x, aug(x)])`` doubling batch/labels

    Because augmentation happens on ``x`` before ``model.forward``, every frontend
    branch (SSL, CQCC/MFCC/LFCC, custom layers, …) sees the **same** augmented audio.
    """

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.concat_original = bool(config.get("concat_original", True))
        self.p = float(config.get("p", 1.0))
        self.mode = str(config.get("mode", "parallel")).lower()
        if self.mode not in {"parallel", "sequential"}:
            raise ValueError(f"batch_augmentation.mode must be parallel or sequential, got {self.mode!r}")

        transforms_cfg = config.get("transforms", [])
        if not transforms_cfg:
            raise ValueError("batch_augmentation.enabled requires a non-empty transforms list")

        self.transforms = []
        for cfg in transforms_cfg:
            cfg_copy = dict(cfg)
            t_type = cfg_copy.pop("type")
            self.transforms.append(build_transform(t_type, **cfg_copy))

        logger.info(
            "Trainer batch augmentation: concat_original=%s mode=%s n_transforms=%d",
            self.concat_original,
            self.mode,
            len(self.transforms),
        )

    def augment_waveforms(self, x: torch.Tensor) -> torch.Tensor:
        """Return augmented waveforms with the same shape/dtype/device as ``x``."""
        if not self.enabled or not self.transforms:
            return x
        if random.random() > self.p:
            return x

        x_np = x.detach().cpu().numpy()
        augmented = []
        for waveform in x_np:
            wav = np.asarray(waveform, dtype=np.float32)
            if self.mode == "parallel":
                transform = random.choice(self.transforms)
                augmented.append(transform(wav.copy()))
            else:
                out = wav.copy()
                for transform in self.transforms:
                    out = transform(out)
                augmented.append(out)

        aug = torch.from_numpy(np.stack(augmented, axis=0))
        return aug.to(device=x.device, dtype=x.dtype)

    def apply_batch(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Augment and optionally concatenate original + augmented (Adapters style)."""
        x_aug = self.augment_waveforms(x)
        if self.concat_original:
            x = torch.cat([x, x_aug], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            if mask is not None:
                mask = torch.cat([mask, mask], dim=0)
        else:
            x = x_aug
        return x, labels, mask
