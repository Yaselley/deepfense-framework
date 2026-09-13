"""Trainer-time multi-resolution crop.

Audio is still loaded and padded to the YAML ``max_len`` (e.g. 4s). During
training, each batch randomly keeps one shorter duration so the forward/loss
run on that resolution only. Validation/test are left at full length.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_DURATIONS_SEC = (0.5, 1.0, 2.0, 3.0, 4.0)


def _to_plain_dict(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "items"):
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(config, resolve=True)
        except Exception:
            return dict(config)
    if isinstance(config, dict):
        return config
    return None


def build_multi_resolution_cropper(config: dict[str, Any] | None) -> MultiResolutionCropper | None:
    cfg = _to_plain_dict(config)
    if not cfg or not cfg.get("enabled", False):
        return None
    return MultiResolutionCropper(cfg)


class MultiResolutionCropper:
    """Crop every sample in a batch to one randomly chosen duration."""

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.sample_rate = int(config.get("sample_rate", 16000))
        if self.sample_rate <= 0:
            raise ValueError(f"multi_resolution.sample_rate must be > 0, got {self.sample_rate}")

        self.random_crop = bool(config.get("random_crop", True))
        self.durations_samples = self._resolve_durations(config)
        weights = config.get("weights")
        if weights is None:
            self.weights = None
        else:
            self.weights = [float(w) for w in weights]
            if len(self.weights) != len(self.durations_samples):
                raise ValueError(
                    "multi_resolution.weights must match the number of durations "
                    f"({len(self.durations_samples)}), got {len(self.weights)}"
                )
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("multi_resolution.weights must be non-negative and not all zero")

        self.last_duration_samples: int | None = None
        self.last_duration_sec: float | None = None

        logger.info(
            "Trainer multi-resolution crop: durations_samples=%s sample_rate=%d random_crop=%s",
            self.durations_samples,
            self.sample_rate,
            self.random_crop,
        )

    def _resolve_durations(self, config: dict[str, Any]) -> list[int]:
        if config.get("durations_samples") is not None:
            durations = [int(n) for n in config["durations_samples"]]
        else:
            raw = config.get("durations_sec", DEFAULT_DURATIONS_SEC)
            durations = [max(1, int(round(float(sec) * self.sample_rate))) for sec in raw]

        if not durations:
            raise ValueError("multi_resolution requires at least one duration")
        if any(n <= 0 for n in durations):
            raise ValueError(f"multi_resolution durations must be > 0 samples, got {durations}")
        return durations

    def sample_duration(self, max_len: int) -> int:
        if self.weights is None:
            target = random.choice(self.durations_samples)
        else:
            target = random.choices(self.durations_samples, weights=self.weights, k=1)[0]
        return min(int(target), int(max_len))

    def apply(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Crop ``x`` (and optional ``mask``) to one duration shared by the batch."""
        if not self.enabled:
            return x, mask

        time_dim = x.shape[-1]
        target_len = self.sample_duration(time_dim)
        self.last_duration_samples = target_len
        self.last_duration_sec = target_len / float(self.sample_rate)

        if target_len >= time_dim:
            return x, mask

        if self.random_crop:
            start = random.randint(0, time_dim - target_len)
        else:
            start = 0
        end = start + target_len

        x = x[..., start:end]
        if mask is not None:
            mask = mask[..., start:end]
        return x, mask
