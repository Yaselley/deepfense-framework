"""SSL + spectral-feature fusion blocks (Multi_View paper cases)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_spectral_frames(
    sf: torch.Tensor,
    target_len: int,
    *,
    method: str = "pool2",
) -> torch.Tensor:
    """
    Align spectral (SF) frames to the SSL frame count.

    XLSR at 16 kHz uses hop 320 (~20 ms) → ~200 frames for 4 s audio.
    Cepstral features (spafe) use ~10 ms hops → ~400 frames for the same clip,
    i.e. roughly 2 SF frames per SSL frame. Multi_View downsamples SF with a
    pairwise mean over consecutive frames before fusion/concat.

    Methods:
      - ``pool2`` (default): ``view(B, T_ssl, 2, D).mean(2)`` when ``T_sf >= 2*T_ssl``
      - ``mean_pool``: integer factor mean-pool when ``T_sf % T_ssl == 0``
      - ``interpolate``: linear resampling fallback
    """
    b, t, d = sf.shape
    if t == target_len:
        return sf

    if method == "pool2" and t >= 2 * target_len:
        sf = sf[:, : target_len * 2, :]
        return sf.view(b, target_len, 2, d).mean(dim=2)

    if method in {"pool2", "mean_pool"} and t > target_len and t % target_len == 0:
        factor = t // target_len
        return sf[:, : target_len * factor, :].view(b, target_len, factor, d).mean(dim=2)

    if method == "interpolate" or method in {"pool2", "mean_pool"}:
        return F.interpolate(
            sf.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    raise ValueError(f"Unknown spectral alignment method: {method!r}")


class GatedFusionBlock(nn.Module):
    """Channel-wise gated fusion between spectral and SSL streams."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, sf: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        sf_c = sf.transpose(1, 2)
        ssl_c = ssl.transpose(1, 2)
        gate = torch.tanh(self.conv2(self.conv1(torch.cat([sf_c, ssl_c], dim=1))))
        fused = (1 - gate) * sf_c + (1 + gate) * ssl_c
        return fused.transpose(1, 2)


class SSLSFFusionHead(nn.Module):
    """
    Fuse projected SSL and spectral streams.

    Modes (paper / Multi_View cases):
      - ``concat``: concatenate then linear projection
      - ``cross_attention``: bidirectional cross-attention then concat
      - ``semi_cross_attention``: SSL queries SF (H_SSL→SF), residual + projection
      - ``expert``: learned per-frame mixture weights
      - ``gated``: gated fusion block
      - ``weighted``: global scalar mixture weights
      - ``ssl_only``: ignore spectral branch
      - ``sf_only``: ignore SSL branch
    """

    def __init__(
        self,
        *,
        fusion_mode: str,
        proj_dim: int = 128,
        spectral_dim: int = 60,
        ssl_attn_heads: int = 8,
        sf_attn_heads: int = 8,
        spectral_align: str = "pool2",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.proj_dim = proj_dim
        self.spectral_dim = spectral_dim
        self.spectral_align = spectral_align

        if fusion_mode not in {
            "concat",
            "cross_attention",
            "semi_cross_attention",
            "expert",
            "gated",
            "weighted",
            "ssl_only",
            "sf_only",
        }:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

        self.projection_ssl = nn.Linear(1024, proj_dim)
        self.projection_sf = nn.Linear(spectral_dim, proj_dim)

        if fusion_mode == "concat":
            self.merge = nn.Linear(proj_dim * 2, proj_dim)
        elif fusion_mode == "cross_attention":
            self.merge = nn.Linear(proj_dim * 2, proj_dim)
            self.attn_ssl = nn.MultiheadAttention(
                proj_dim, num_heads=ssl_attn_heads, batch_first=True
            )
            self.attn_sf = nn.MultiheadAttention(
                proj_dim, num_heads=sf_attn_heads, batch_first=True
            )
        elif fusion_mode == "semi_cross_attention":
            self.merge = nn.Linear(proj_dim, proj_dim)
            self.attn_ssl = nn.MultiheadAttention(
                proj_dim, num_heads=ssl_attn_heads, batch_first=True
            )
        elif fusion_mode == "expert":
            self.gate = nn.Linear(proj_dim, 2)
        elif fusion_mode == "gated":
            self.fuser = GatedFusionBlock(proj_dim)
        elif fusion_mode == "weighted":
            self.raw_weights = nn.Parameter(torch.ones(2))
        elif fusion_mode == "sf_only":
            self.merge = nn.Identity()
        elif fusion_mode == "ssl_only":
            self.merge = nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.proj_dim

    def _project_streams(
        self,
        ssl: torch.Tensor,
        sf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ssl_proj = self.projection_ssl(ssl)
        if sf is None:
            return ssl_proj, None
        sf_proj = self.projection_sf(sf)
        sf_proj = align_spectral_frames(
            sf_proj,
            ssl_proj.shape[1],
            method=self.spectral_align,
        )
        return ssl_proj, sf_proj

    def forward(self, ssl: torch.Tensor | None, sf: torch.Tensor | None) -> torch.Tensor:
        if self.fusion_mode == "ssl_only":
            if ssl is None:
                raise ValueError("ssl_only fusion requires SSL features")
            return self.projection_ssl(ssl)
        if self.fusion_mode == "sf_only":
            if sf is None:
                raise ValueError("sf_only fusion requires spectral features")
            sf_proj = self.projection_sf(sf)
            b, t, d = sf_proj.shape
            if t >= 2:
                t_even = t - (t % 2)
                sf_proj = sf_proj[:, :t_even, :].view(b, t_even // 2, 2, d).mean(dim=2)
            return sf_proj

        if sf is None:
            raise ValueError(f"fusion_mode={self.fusion_mode!r} requires spectral features")

        ssl_proj, sf_proj = self._project_streams(ssl, sf)

        if self.fusion_mode == "concat":
            return self.merge(torch.cat([ssl_proj, sf_proj], dim=-1))

        if self.fusion_mode == "cross_attention":
            ssl_attn, _ = self.attn_ssl(query=ssl_proj, key=sf_proj, value=sf_proj)
            sf_attn, _ = self.attn_sf(query=sf_proj, key=ssl_proj, value=ssl_proj)
            fused = torch.cat([ssl_proj + ssl_attn, sf_proj + sf_attn], dim=-1)
            return self.merge(fused)

        if self.fusion_mode == "semi_cross_attention":
            ssl_attn, _ = self.attn_ssl(query=ssl_proj, key=sf_proj, value=sf_proj)
            return self.merge(ssl_proj + ssl_attn)

        if self.fusion_mode == "expert":
            logits = self.gate(ssl_proj)
            weights = F.softmax(logits, dim=-1)
            w_sf, w_ssl = weights.split(1, dim=-1)
            return w_sf * sf_proj + w_ssl * ssl_proj

        if self.fusion_mode == "gated":
            return self.fuser(sf_proj, ssl_proj)

        if self.fusion_mode == "weighted":
            w_sf, w_ssl = F.softmax(self.raw_weights, dim=0)
            return w_sf * sf_proj + w_ssl * ssl_proj

        raise RuntimeError(f"Unhandled fusion_mode: {self.fusion_mode}")
