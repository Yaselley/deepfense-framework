"""Optional debug tracing for the temporal / partial-deepfake pipeline."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import torch

_LOGGER = logging.getLogger("deepfense.trace")


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


_ENABLED = _env_flag("DEEPFENSE_TRACE", False)
_LIMIT = _env_int("DEEPFENSE_TRACE_LIMIT", 3)

_counts: dict[str, int] = {}
_lock = threading.Lock()


def is_enabled() -> bool:
    return _ENABLED and _LIMIT != 0


def should_trace(key: str) -> bool:
    if not _ENABLED or _LIMIT == 0:
        return False
    if _LIMIT < 0:
        return True
    with _lock:
        n = _counts.get(key, 0)
        if n >= _LIMIT:
            return False
        _counts[key] = n + 1
        return True


def reset_counters() -> None:
    with _lock:
        _counts.clear()


def t_summary(name: str, x: Any, sample: int = 8) -> str:
    try:
        return _t_summary_impl(name, x, sample)
    except Exception as exc:  # noqa: BLE001
        return f"{name}=<t_summary failed: {type(exc).__name__}: {exc}>"


def _t_summary_impl(name: str, x: Any, sample: int = 8) -> str:
    if x is None:
        return f"{name}=None"
    if isinstance(x, torch.Tensor):
        shape = tuple(x.shape)
        s = f"{name}: torch.Tensor shape={shape} dtype={x.dtype} device={x.device}"
        if x.numel() == 0:
            return s + " <empty>"
        with torch.no_grad():
            xf = x.detach()
            if xf.dtype.is_floating_point:
                s += (
                    f" min={xf.min().item():.4g} max={xf.max().item():.4g}"
                    f" mean={xf.float().mean().item():.4g}"
                )
            else:
                flat = xf.reshape(-1)
                if flat.numel() <= 100_000:
                    uniq, cnts = torch.unique(flat, return_counts=True)
                    pairs = ", ".join(
                        f"{int(u)}:{int(c)}"
                        for u, c in zip(uniq.tolist(), cnts.tolist())
                    )
                    s += f" unique={{{pairs}}}"
            head = xf.reshape(-1)[:sample].tolist()
            s += f" head={head}"
        return s
    if isinstance(x, np.ndarray):
        shape = tuple(x.shape)
        s = f"{name}: np.ndarray shape={shape} dtype={x.dtype}"
        if x.size == 0:
            return s + " <empty>"
        if np.issubdtype(x.dtype, np.floating):
            s += (
                f" min={float(x.min()):.4g} max={float(x.max()):.4g}"
                f" mean={float(x.mean()):.4g}"
            )
        else:
            if x.size <= 100_000:
                uniq, cnts = np.unique(x, return_counts=True)
                pairs = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq.tolist(), cnts.tolist()))
                s += f" unique={{{pairs}}}"
        s += f" head={x.reshape(-1)[:sample].tolist()}"
        return s
    return f"{name}={x!r} ({type(x).__name__})"


def trace(key: str, msg: str, *args: Any) -> None:
    if not should_trace(key):
        return
    _LOGGER.info("[TRACE %s] " + msg, key, *args)


def trace_block(key: str, title: str, lines: list[str]) -> None:
    if not should_trace(key):
        return
    _LOGGER.info("[TRACE %s] %s", key, title)
    for line in lines:
        _LOGGER.info("[TRACE %s]   %s", key, line)
