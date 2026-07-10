"""Save clip-level and frame-level evaluation predictions as JSONL."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# store[dataset_name][utt_id] -> utterance record with list fields
FramePredictionStore = Dict[str, Dict[str, dict]]


def resolve_label_hop(data_cfg: dict, sampling_rate: float = 16000.0) -> Dict[str, Optional[float]]:
    """Resolve label hop in ms and samples from a data config block."""
    label_hop_ms = None
    label_hop_samples = None

    if data_cfg.get("label_hop_ms") is not None:
        label_hop_ms = float(data_cfg["label_hop_ms"])
    elif data_cfg.get("label_hop") is not None:
        label_hop_samples = int(data_cfg["label_hop"])
        label_hop_ms = label_hop_samples * 1000.0 / float(sampling_rate)

    if label_hop_samples is None and label_hop_ms is not None:
        label_hop_samples = int(round(label_hop_ms * float(sampling_rate) / 1000.0))

    return {
        "label_hop_ms": label_hop_ms,
        "label_hop_samples": label_hop_samples,
        "sampling_rate": float(sampling_rate),
    }


def build_eval_metadata(
    temporal: bool,
    label_hop_ms: Optional[float] = None,
    label_hop_samples: Optional[int] = None,
    sampling_rate: Optional[float] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"temporal": bool(temporal)}
    if label_hop_ms is not None:
        meta["label_hop_ms"] = float(label_hop_ms)
    if label_hop_samples is not None:
        meta["label_hop_samples"] = int(label_hop_samples)
    if sampling_rate is not None:
        meta["sampling_rate"] = float(sampling_rate)
    return meta


def _utterance_record(
    utt_id: str,
    label_hop_ms: float,
    label_hop_samples: Optional[int],
) -> dict:
    rec = {
        "ID": utt_id,
        "label_hop_ms": float(label_hop_ms),
        "frame_idx": [],
        "time_sec": [],
        "label": [],
        "score_class0": [],
        "score_class1": [],
        "score_llr": [],
    }
    if label_hop_samples is not None:
        rec["label_hop_samples"] = int(label_hop_samples)
    return rec


def append_frame_predictions(
    store: FramePredictionStore,
    *,
    utt_ids: Sequence[str],
    dataset_names: Sequence[str],
    frame_labels: np.ndarray,
    logits: np.ndarray,
    valid_mask: np.ndarray,
    label_hop_ms: float,
    label_hop_samples: Optional[int],
    bonafide_label: int,
) -> None:
    """Append framewise scores into per-utterance list fields."""
    hop_sec = float(label_hop_ms) / 1000.0
    bona = int(bonafide_label)
    spoof = abs(1 - bona)
    batch_size = frame_labels.shape[0]

    for i in range(batch_size):
        ds = dataset_names[i]
        utt_id = utt_ids[i]
        ds_store = store.setdefault(ds, {})
        if utt_id not in ds_store:
            ds_store[utt_id] = _utterance_record(utt_id, label_hop_ms, label_hop_samples)
        rec = ds_store[utt_id]

        for fi in np.where(valid_mask[i])[0]:
            fi = int(fi)
            rec["frame_idx"].append(fi)
            rec["time_sec"].append(round(fi * hop_sec, 6))
            rec["label"].append(int(frame_labels[i, fi]))
            rec["score_class0"].append(float(logits[i, fi, 0]))
            rec["score_class1"].append(float(logits[i, fi, 1]))
            rec["score_llr"].append(float(logits[i, fi, bona] - logits[i, fi, spoof]))


def save_jsonl(path: str, records: Sequence[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def save_frame_predictions_jsonl(
    path: str,
    utterances: Dict[str, dict],
    metadata: Optional[dict] = None,
) -> None:
    """Write one JSON object per utterance; optional metadata line first."""
    lines: List[dict] = []
    if metadata:
        lines.append({"type": "metadata", **metadata})
    for utt_id in sorted(utterances.keys()):
        lines.append(utterances[utt_id])
    save_jsonl(path, lines)


def save_clip_predictions_jsonl(
    path: str,
    keys: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
    metadata: Optional[dict] = None,
) -> None:
    """Write one JSON object per clip (utterance-level evaluation)."""
    if scores.ndim == 1:
        scores_c1 = scores
        scores_c0 = 1.0 - scores
    elif scores.ndim == 2 and scores.shape[1] == 2:
        scores_c0 = scores[:, 0]
        scores_c1 = scores[:, 1]
    elif scores.ndim == 2 and scores.shape[1] == 1:
        scores_c1 = scores.flatten()
        scores_c0 = 1.0 - scores_c1
    else:
        raise ValueError(f"Unsupported clip score shape: {scores.shape}")

    lines: List[dict] = []
    if metadata:
        lines.append({"type": "metadata", **metadata})
    for i in range(len(labels)):
        lines.append(
            {
                "ID": keys[i],
                "label": int(labels[i]),
                "score_class0": float(scores_c0[i]),
                "score_class1": float(scores_c1[i]),
            }
        )
    save_jsonl(path, lines)
