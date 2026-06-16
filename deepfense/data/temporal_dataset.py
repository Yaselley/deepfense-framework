"""Per-frame (partial deepfake) dataset with dense ``frame_labels`` in parquet."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from deepfense.data.base_dataset import BaseDataset
from deepfense.data.temporal_utils import (
    approx_num_frames,
    dense_labels_to_fixed,
    downsample_frame_labels,
    mask_trailing_frames,
)
from deepfense.data.transforms.transforms import load_audio
from deepfense.utils.registry import register_dataset, build_transforms_pipeline

logger = logging.getLogger(__name__)

_IGNORE = -100

_NULLISH = frozenset({"none", "null", ""})


def _optional_int(value: Any) -> int | None:
    """Parse optional int config values; treat YAML/OmegaConf nullish strings as None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _NULLISH:
            return None
        return int(s)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(value)


def _parse_label_vector(raw: Any) -> np.ndarray:
    if raw is None:
        return np.zeros((0,), dtype=np.int64)
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    if isinstance(raw, str):
        s = raw.strip()
        if s.endswith(".npy") or s.endswith(".npz"):
            raise ValueError("Use frame_labels_path column for file paths, not frame_labels string.")
        if s.startswith("[") and s.endswith("]"):
            return np.asarray(json.loads(s), dtype=np.int64).reshape(-1)
        return np.asarray([int(x) for x in s.split(",") if x.strip() != ""], dtype=np.int64)
    return np.asarray(raw, dtype=np.int64).reshape(-1)


def _load_labels_from_path(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"frame_labels file not found: {path}")
    if path.endswith(".npz"):
        z = np.load(path)
        arr = z[z.files[0]]
    else:
        arr = np.load(path, allow_pickle=False)
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def _parquet_cell_not_missing(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, np.ndarray):
        if val.dtype == np.dtype("O") and val.ndim == 0:
            return _parquet_cell_not_missing(val.item())
        return val.size > 0 and bool(np.any(pd.notna(val)))
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return False
        return bool(np.any(pd.notna(np.asarray(val, dtype=object))))
    if isinstance(val, str):
        return pd.notna(val) and len(val.strip()) > 0
    return bool(pd.notna(val))


def _strip_cropping_transforms(base_cfg):
    """Force pad transforms to non-cropping mode for full-audio partial-deepfake clips."""
    if not base_cfg:
        return base_cfg

    out = []
    rewritten = False
    for t in base_cfg:
        if isinstance(t, dict) and t.get("type") == "pad":
            t_new = dict(t)
            had_max = t_new.get("max_len") is not None
            had_truncate = bool(t_new.get("truncate", True))
            if had_max or had_truncate:
                rewritten = True
            t_new["max_len"] = None
            t_new["truncate"] = False
            out.append(t_new)
        else:
            out.append(t)

    if rewritten:
        logger.warning(
            "TemporalSegmentationDataset is forcing 'pad' transforms to "
            "max_len=None, truncate=False (partial-deepfake clips must not be "
            "cropped). Variable-length batches are zero-padded by collate_fn."
        )
    return out


@register_dataset("TemporalSegmentationDataset")
class TemporalSegmentationDataset(BaseDataset):
    """
    Per-frame binary classification dataset for partial deepfake detection.

    Parquet columns (one row per utterance):
        path (required): audio file path.
        frame_labels OR frame_labels_path (one required per row):
            - frame_labels: inline labels — Python list (recommended), numpy
              array, comma-separated string (``"0,1,0,1"``), or JSON list
              string (``"[0, 1, 0, 1]"``). Do not put ``.npy``/``.npz`` paths here.
            - frame_labels_path: path to a 1D ``.npy`` or ``.npz`` label file.
        If both columns are set on a row, ``frame_labels_path`` is used.
        label (optional): clip-level class; inferred from frames if omitted.
        ID (optional): utterance id.

    Label values are integer class indices (``label_map``). Length = one frame
    per ``source_label_hop`` samples (see ``label_hop`` / ``source_label_hop``).

    Args:
        cfg (dict): Configuration with keys:
            - parquet_files (list[str]): Paths to parquet metadata files
            - label_map (dict): Mapping from label strings to integers
            - frame_labels or frame_labels_path: per-frame labels at source_label_hop
            - label_hop / label_hop_ms: training prediction rate in samples or ms
            - source_label_hop / source_label_hop_ms: annotation rate (defaults to label_hop)
            - label_merge_rule: any_spoof | all_spoof | majority | any_non_bonafide
            - root_dir (str, optional): Base directory to prepend to paths in parquet
            - dataset_names (list[str], optional): Names for each parquet file
            - max_per_class (int, optional): Maximum samples per class
            - sampling_rate (int): Audio sample rate (default: 16000)
            - base_transform (list): Base transform pipeline config
            - augment_transform (list): Augmentation pipeline config
    """

    def __init__(self, cfg):
        super().__init__()
        self.config_data = cfg
        self.label_map = self.config_data["label_map"]
        self.parquet_files = self.config_data["parquet_files"]
        self.dataset_names = self.config_data.get("dataset_names", None)
        self.root_dir = self.config_data.get("root_dir", None)
        self.max_per_class = self.config_data.get("max_per_class", None)
        self.sampling_rate = int(self.config_data.get("sampling_rate", 16000))

        self.label_hop = self._resolve_label_hop()
        self.source_label_hop = self._resolve_source_label_hop()
        self.label_merge_rule = str(
            self.config_data.get("label_merge_rule", "any_spoof")
        ).lower()
        self._spoof_int = int(self.label_map.get("spoof", 0))
        self._bonafide_int = int(self.label_map.get("bonafide", 1))

        if self.label_hop % self.source_label_hop != 0:
            raise ValueError(
                f"label_hop ({self.label_hop}) must be an integer multiple of "
                f"source_label_hop ({self.source_label_hop}); got ratio "
                f"{self.label_hop / self.source_label_hop:.3f}. Re-annotate or "
                f"pick a label_hop divisible by your annotation rate."
            )
        self._label_pool_factor = self.label_hop // self.source_label_hop
        if self._label_pool_factor < 1:
            raise ValueError(
                f"label_hop ({self.label_hop}) is finer than source_label_hop "
                f"({self.source_label_hop}); upsampling labels would invent "
                f"information. Pick label_hop >= source_label_hop."
            )

        if self.config_data.get("full_audio") is False:
            logger.warning(
                "TemporalSegmentationDataset ignores full_audio=False -- "
                "partial-deepfake training requires the full waveform per clip."
            )

        # Optional synchronized crop: slice audio + labels with the same offset.
        # Shorter clips are returned as-is; collate_fn pads the batch.
        self.crop_max_len = _optional_int(self.config_data.get("max_len"))
        self.random_crop = bool(self.config_data.get("random_crop", False))

        self.max_frames_cap = _optional_int(self.config_data.get("max_frames", None))

        self.random_pad = False

        self.base_transform_cfg = _strip_cropping_transforms(
            self.config_data.get("base_transform", None)
        )
        self.augment_transform_cfg = self.config_data.get("augment_transform", None)
        self.base_transform = build_transforms_pipeline(self.base_transform_cfg)
        self.augment_transform = build_transforms_pipeline(self.augment_transform_cfg)

        logger.info(
            "\n"
            "  TemporalSegmentationDataset\n"
            "  ├─ sampling_rate      : %d Hz\n"
            "  ├─ label_hop          : %d samples  (%.1f ms)\n"
            "  ├─ source_label_hop   : %d samples  (%.1f ms)\n"
            "  ├─ label_merge_rule   : %s\n"
            "  ├─ max_len (crop)     : %s\n"
            "  ├─ random_crop        : %s\n"
            "  └─ max_frames_cap     : %s",
            self.sampling_rate,
            self.label_hop,      1000.0 * self.label_hop / self.sampling_rate,
            self.source_label_hop, 1000.0 * self.source_label_hop / self.sampling_rate,
            self.label_merge_rule,
            f"{self.crop_max_len} samples ({1000.0 * self.crop_max_len / self.sampling_rate:.1f} ms)"
                if self.crop_max_len else "None (full clip)",
            self.random_crop,
            self.max_frames_cap,
        )

        self.data = []
        for i, p_file in enumerate(self.parquet_files):
            if not os.path.exists(p_file):
                raise FileNotFoundError(f"Parquet file not found: {p_file}")
            df = pd.read_parquet(p_file)
            if len(df) == 0:
                raise ValueError(f"Parquet file is empty: {p_file}")

            has_vec = "frame_labels" in df.columns
            has_path = "frame_labels_path" in df.columns
            if not (has_vec or has_path):
                raise ValueError(
                    f"Parquet '{p_file}' needs 'frame_labels' and/or 'frame_labels_path'."
                )
            req = ["path"]
            miss = [c for c in req if c not in df.columns]
            if miss:
                raise ValueError(f"Parquet '{p_file}' missing columns: {miss}")

            df["dataset_name"] = (
                self.dataset_names[i] if self.dataset_names and i < len(self.dataset_names) else f"dataset_{i}"
            )
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)

        if "label" in self.data.columns:
            self.data["label"] = self.data["label"].map(self.label_map)
        else:
            spoof_val = int(self.label_map.get("spoof", 0))
            bonafide_val = int(self.label_map.get("bonafide", 1))

            def _weak_label(row):
                fl = self._get_frame_labels_array(row)
                if fl.size == 0:
                    return bonafide_val
                return int(spoof_val) if np.any(fl == spoof_val) else int(bonafide_val)

            self.data["label"] = self.data.apply(_weak_label, axis=1)

        if self.max_per_class is not None:
            limited = []
            for lbl in self.data["label"].unique():
                sub = self.data[self.data["label"] == lbl]
                limited.append(sub.sample(n=min(self.max_per_class, len(sub))))
            self.data = pd.concat(limited, ignore_index=True)

    def _resolve_label_hop(self) -> int:
        if "label_hop" in self.config_data and self.config_data.get("label_hop") is not None:
            hop = int(self.config_data["label_hop"])
        elif "label_hop_ms" in self.config_data and self.config_data.get("label_hop_ms") is not None:
            hop_ms = float(self.config_data["label_hop_ms"])
            hop = int(round(hop_ms * self.sampling_rate / 1000.0))
        elif "subsample" in self.config_data and self.config_data.get("subsample") is not None:
            logger.warning(
                "TemporalSegmentationDataset: 'subsample' is deprecated; "
                "use 'label_hop' (samples) or 'label_hop_ms' (ms). "
                "Treating subsample=%s as label_hop.",
                self.config_data["subsample"],
            )
            hop = int(self.config_data["subsample"])
        else:
            hop = 320

        if hop <= 0:
            raise ValueError(f"label_hop must be a positive integer, got {hop}")
        return hop

    def _resolve_source_label_hop(self) -> int:
        if (
            "source_label_hop" in self.config_data
            and self.config_data.get("source_label_hop") is not None
        ):
            hop = int(self.config_data["source_label_hop"])
        elif (
            "source_label_hop_ms" in self.config_data
            and self.config_data.get("source_label_hop_ms") is not None
        ):
            hop = int(round(
                float(self.config_data["source_label_hop_ms"])
                * self.sampling_rate / 1000.0
            ))
        else:
            hop = self.label_hop
        if hop <= 0:
            raise ValueError(f"source_label_hop must be positive, got {hop}")
        return hop

    def _get_audio_path(self, path):
        if self.root_dir is not None:
            return os.path.join(self.root_dir, path[1:])
        return path

    def _get_frame_labels_array(self, row) -> np.ndarray:
        if "frame_labels_path" in row.index and _parquet_cell_not_missing(row["frame_labels_path"]):
            p = row["frame_labels_path"]
            path = self._get_audio_path(str(p)) if not os.path.isabs(str(p)) else str(p)
            return _load_labels_from_path(path)
        if "frame_labels" in row.index and _parquet_cell_not_missing(row["frame_labels"]):
            return _parse_label_vector(row["frame_labels"])
        return np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = self._get_audio_path(row["path"])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        sample_id = row.get("ID", f"{row['dataset_name']}_{idx}")

        x_raw = load_audio(
            path=audio_path,
            target_sr=self.config_data.get("target_sr", self.sampling_rate),
            mono=self.config_data.get("mono", True),
        )
        n_original = int(x_raw.shape[0])
        fl_raw = self._get_frame_labels_array(row)

        # Synchronized crop: slice audio and labels with the same sample offset.
        # Shorter clips are returned as-is; collate_fn handles batch padding.
        if self.crop_max_len is not None and n_original > self.crop_max_len:
            if self.random_crop:
                max_start = n_original - self.crop_max_len
                start = np.random.randint(0, max_start + 1)
                start = (start // self.label_hop) * self.label_hop
            else:
                start = 0
            x_raw = x_raw[start : start + self.crop_max_len]
            n_original = int(x_raw.shape[0])
            src_start = start // self.source_label_hop
            src_len   = self.crop_max_len // self.source_label_hop
            fl_raw = fl_raw[src_start : src_start + src_len]

        if self.base_transform:
            x = self.base_transform(x_raw)
        else:
            x = x_raw
        if self.augment_transform:
            x = self.augment_transform(x)

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            n_eff = min(n_original, int(x.shape[0]))
            xt = torch.tensor(x, dtype=torch.float32)
        elif x.ndim == 2:
            n_eff = min(n_original, int(x.shape[1]))
            xt = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 1D or 2D waveform, got shape {x.shape}")

        fl = fl_raw
        source_audio_frames = approx_num_frames(n_eff, self.source_label_hop)
        if fl.size > 0 and abs(int(fl.size) - int(source_audio_frames)) > 1:
            logger.warning(
                "Row %s: frame_labels length=%d but audio gives %d frames at "
                "source_label_hop=%d. Mismatch will be silently aligned.",
                row.get("ID", idx),
                int(fl.size),
                int(source_audio_frames),
                self.source_label_hop,
            )

        if self._label_pool_factor > 1:
            fl = downsample_frame_labels(
                fl,
                factor=self._label_pool_factor,
                rule=self.label_merge_rule,
                spoof_label=self._spoof_int,
                bonafide_label=self._bonafide_int,
                ignore_value=_IGNORE,
            )

        audio_frames = approx_num_frames(n_eff, self.label_hop)
        target_frames = audio_frames
        if self.max_frames_cap is not None:
            target_frames = min(target_frames, self.max_frames_cap)

        labels_fixed = dense_labels_to_fixed(fl, target_frames, pad_value=_IGNORE)
        valid_frames = min(audio_frames, target_frames)
        labels_fixed = mask_trailing_frames(labels_fixed, valid_frames, pad_value=_IGNORE)

        clip_label = int(row["label"])

        return {
            "ID": sample_id,
            "x": xt,
            "label": torch.tensor(clip_label, dtype=torch.long),
            "frame_labels": torch.tensor(labels_fixed, dtype=torch.long),
            "dataset_name": row["dataset_name"],
        }
