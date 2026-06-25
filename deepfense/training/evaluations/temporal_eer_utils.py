"""Shared helpers for framewise localization metrics (Segment-EER, Range-EER)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from deepfense.training.evaluations.compute_eer import _det_curve_for_eer
from deepfense.training.evaluations.utils import _metric_get_1d_scores

Segment = Tuple[float, float]


def _compute_rate(num: float, denom: float) -> float:
    if denom == 0.0:
        return 0.0 if num == 0.0 else 1.0
    return num / denom


def group_frames_by_utterance(
    keys: Sequence,
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Group flattened frame labels/scores by utterance id."""
    groups: Dict[str, Dict[str, list]] = defaultdict(lambda: {"labels": [], "scores": []})
    for key, label, score in zip(keys, labels, scores):
        groups[str(key)]["labels"].append(label)
        groups[str(key)]["scores"].append(score)
    return {
        utt: (np.asarray(v["labels"], dtype=np.int64), np.asarray(v["scores"], dtype=float))
        for utt, v in groups.items()
    }


def binary_mask_to_segments(mask: np.ndarray, hop_sec: float) -> List[Segment]:
    """Convert a boolean frame mask to contiguous time segments in seconds."""
    segments: List[Segment] = []
    if mask.size == 0:
        return segments

    start_idx = None
    for idx, active in enumerate(mask):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            segments.append((start_idx * hop_sec, idx * hop_sec))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx * hop_sec, mask.size * hop_sec))
    return segments


def _segment_duration(segments: Iterable[Segment]) -> float:
    return float(sum(max(0.0, end - start) for start, end in segments))


def _intersect_duration(a: Sequence[Segment], b: Sequence[Segment]) -> float:
    total = 0.0
    for a_start, a_end in a:
        for b_start, b_end in b:
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if end > start:
                total += end - start
    return total


def spoof_segments_from_frames(
    frame_labels: np.ndarray,
    hop_sec: float,
    spoof_label: int,
    ignore_index: int = -100,
) -> List[Segment]:
    valid = frame_labels != ignore_index
    spoof_mask = valid & (frame_labels == spoof_label)
    return binary_mask_to_segments(spoof_mask, hop_sec)


def spoof_segments_from_scores(
    scores: np.ndarray,
    threshold: float,
    hop_sec: float,
) -> List[Segment]:
    """Higher score = bonafide; spoof where score <= threshold."""
    spoof_mask = scores <= threshold
    return binary_mask_to_segments(spoof_mask, hop_sec)


def detection_error_rates(
    ref_spoof: Sequence[Segment],
    hyp_spoof: Sequence[Segment],
    duration_sec: float,
) -> Tuple[float, float]:
    """
    Range-based detection error rates (PartialSpoof / pyannote-style).

    Positive class = spoof segments in the reference.
    """
    duration_sec = max(float(duration_sec), 0.0)
    ref_spoof_dur = _segment_duration(ref_spoof)
    ref_bona_dur = max(duration_sec - ref_spoof_dur, 0.0)

    overlap = _intersect_duration(ref_spoof, hyp_spoof)
    miss_dur = max(ref_spoof_dur - overlap, 0.0)
    hyp_spoof_dur = _segment_duration(hyp_spoof)
    fa_dur = max(hyp_spoof_dur - overlap, 0.0)

    fnr = _compute_rate(miss_dur, ref_spoof_dur)
    fpr = _compute_rate(fa_dur, ref_bona_dur)
    return fpr, fnr


def aggregate_detection_rates(
    utterances: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float,
    hop_sec: float,
    spoof_label: int,
    ignore_index: int = -100,
) -> Tuple[float, float]:
    """Accumulate false-alarm and miss durations across utterances."""
    fa_dur = 0.0
    miss_dur = 0.0
    ref_bona_dur = 0.0
    ref_spoof_dur = 0.0

    for frame_labels, scores in utterances.values():
        valid = frame_labels != ignore_index
        if not np.any(valid):
            continue

        labels = frame_labels[valid]
        utt_scores = scores[valid]
        n_frames = labels.size
        duration_sec = n_frames * hop_sec

        ref_spoof = spoof_segments_from_frames(labels, hop_sec, spoof_label, ignore_index)
        hyp_spoof = spoof_segments_from_scores(utt_scores, threshold, hop_sec)

        ref_spoof_d = _segment_duration(ref_spoof)
        ref_bona_d = max(duration_sec - ref_spoof_d, 0.0)
        overlap = _intersect_duration(ref_spoof, hyp_spoof)

        ref_spoof_dur += ref_spoof_d
        ref_bona_dur += ref_bona_d
        miss_dur += max(ref_spoof_d - overlap, 0.0)
        fa_dur += max(_segment_duration(hyp_spoof) - overlap, 0.0)

    fpr = _compute_rate(fa_dur, ref_bona_dur)
    fnr = _compute_rate(miss_dur, ref_spoof_dur)
    return fpr, fnr


def frame_eer_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    bonafide_label: int = 1,
) -> Tuple[float, float]:
    """Segment/frame-level EER used to initialise Range-EER search."""
    frr, far, thresholds = _det_curve_for_eer(labels, scores, bonafide_label)
    abs_diffs = np.abs(frr - far)
    min_index = int(np.argmin(abs_diffs))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


def utterances_for_range_eer(
    keys: Sequence | None,
    labels: np.ndarray,
    scores: np.ndarray,
    ignore_index: int = -100,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Group valid frames by utterance for range-based detection rates."""
    valid = labels != ignore_index
    labels = np.asarray(labels[valid], dtype=np.int64)
    scores = np.asarray(scores[valid], dtype=np.float64)
    if labels.size == 0:
        return {}
    if keys is None:
        return {"": (labels, scores)}
    keys = np.asarray(keys)[valid]
    return group_frames_by_utterance(keys, labels, scores)


def resolve_range_hop_sec(params: dict) -> float:
    """
    Hop (seconds) for range-based overlap measurement.

    Uses ``source_label_hop_ms`` / ``reference_ms`` when available. When the
    reference hop is finer than eval frames (``label_hop_ms``), eval frames are
    used because fine ground truth is not retained at inference.
    """
    pred_ms = float(params.get("source_ms") or params.get("label_hop_ms") or 20.0)
    ref_ms = float(
        params.get("source_label_hop_ms")
        or params.get("reference_ms")
        or pred_ms
    )
    if ref_ms < pred_ms:
        logger.debug(
            "Range-EER: reference hop %.1f ms is finer than eval frames %.1f ms; "
            "using eval hop.",
            ref_ms,
            pred_ms,
        )
        ref_ms = pred_ms
    return ref_ms / 1000.0


def range_detection_rates(
    utterances: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float,
    hop_sec: float,
    spoof_label: int,
    ignore_index: int = -100,
) -> Tuple[float, float]:
    """Duration-overlap FPR/FNR (PartialSpoof RangeEER / pyannote-style)."""
    if not utterances:
        return float("nan"), float("nan")
    return aggregate_detection_rates(
        utterances, threshold, hop_sec, spoof_label, ignore_index
    )


def search_range_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    keys: Sequence | None = None,
    hop_sec: float,
    spoof_label: int = 0,
    bonafide_label: int = 1,
    ignore_index: int = -100,
    prec: float = 1e-4,
    max_iters: int = 64,
) -> Tuple[float, float]:
    """
    Binary-search threshold so FPR ~= FNR (Range-EER, Zhang et al., INTERSPEECH 2023).

    FPR/FNR are computed from **duration overlap** between reference spoof segments
    (ground-truth labels) and hypothesis spoof segments (score <= threshold), pooled
    per utterance via :func:`aggregate_detection_rates`.
    """
    utterances = utterances_for_range_eer(keys, labels, scores, ignore_index)
    if not utterances:
        return float("nan"), float("nan")

    flat_labels = np.concatenate([u[0] for u in utterances.values()])
    flat_scores = np.concatenate([u[1] for u in utterances.values()])
    if flat_labels.size == 0:
        return float("nan"), float("nan")
    if len(np.unique(flat_labels)) < 2:
        return float("nan"), float("nan")

    all_sorted = np.sort(flat_scores)

    th_lo = float(all_sorted[0])
    th_hi = float(all_sorted[-1])
    lo_p, hi_p = 0.0, 100.0
    th_mid = float(np.percentile(all_sorted, 50.0))

    fpr_lo, fnr_lo = range_detection_rates(
        utterances, th_lo, hop_sec, spoof_label, ignore_index
    )
    fpr_mid, fnr_mid = range_detection_rates(
        utterances, th_mid, hop_sec, spoof_label, ignore_index
    )
    best = (abs(fpr_mid - fnr_mid), th_mid, fpr_mid, fnr_mid)

    for _ in range(max_iters):
        if th_lo >= th_hi or abs(fpr_mid - fnr_mid) <= prec:
            break
        if (fpr_lo - fnr_lo) * (fpr_mid - fnr_mid) < 0:
            th_hi = th_mid
            hi_p = (lo_p + hi_p) / 2.0
        else:
            th_lo = th_mid
            lo_p = (lo_p + hi_p) / 2.0
            fpr_lo, fnr_lo = fpr_mid, fnr_mid
        mid_p = (lo_p + hi_p) / 2.0
        th_mid = float(np.percentile(all_sorted, mid_p))
        fpr_mid, fnr_mid = range_detection_rates(
            utterances, th_mid, hop_sec, spoof_label, ignore_index
        )
        cand = (abs(fpr_mid - fnr_mid), th_mid, fpr_mid, fnr_mid)
        if cand[0] <= best[0]:
            best = cand

    _, th_best, fpr_best, fnr_best = best
    return float(np.mean((fpr_best, fnr_best))), float(th_best)


def pool_utterance_scores(
    scores: np.ndarray,
    pool: str = "min",
) -> float:
    """Aggregate framewise LLR scores to one utterance score (PartialSpoof uses min)."""
    if scores.size == 0:
        return float("nan")
    pool = pool.lower()
    if pool == "min":
        return float(np.min(scores))
    if pool == "max":
        return float(np.max(scores))
    if pool in ("mean", "avg", "average"):
        return float(np.mean(scores))
    raise ValueError(f"Unknown utterance score pooling: {pool!r}")


def infer_utterance_label(
    frame_labels: np.ndarray,
    bonafide_label: int = 1,
    ignore_index: int = -100,
) -> int:
    """Utterance is spoof if any valid frame is spoof (min over frame labels)."""
    valid = frame_labels != ignore_index
    if not np.any(valid):
        return bonafide_label
    spoof_label = abs(1 - bonafide_label)
    if np.any(frame_labels[valid] == spoof_label):
        return spoof_label
    return bonafide_label


def pool_frames_to_utterances(
    keys: Sequence,
    labels: np.ndarray,
    scores: np.ndarray,
    bonafide_label: int = 1,
    pool: str = "min",
    ignore_index: int = -100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-frame data to one score/label per utterance.

    Mirrors PartialSpoof ``get_utteer_by_seg``: ``groupby(Wavid).min()``.
    """
    utterances = group_frames_by_utterance(keys, labels, scores)
    utt_labels = []
    utt_scores = []
    for frame_labels, frame_scores in utterances.values():
        valid = frame_labels != ignore_index
        if not np.any(valid):
            continue
        fl = frame_labels[valid]
        fs = frame_scores[valid]
        utt_labels.append(infer_utterance_label(fl, bonafide_label, ignore_index))
        utt_scores.append(pool_utterance_scores(fs, pool=pool))
    return (
        np.asarray(utt_labels, dtype=int),
        np.asarray(utt_scores, dtype=float),
    )


def prepare_temporal_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure 1D LLR scores for temporal metrics."""
    scores_1d = _metric_get_1d_scores(scores, params)
    return np.asarray(labels).astype(int), np.asarray(scores_1d, dtype=float)


def normalize_resolutions_ms(value) -> List[float]:
    """Accept a single ms value or a list, e.g. ``20`` or ``[20, 40, 80]``."""
    if value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    return [float(x) for x in value]


def _pool_axis(values: np.ndarray, pool: str) -> np.ndarray:
    pool = pool.lower()
    if pool == "min":
        return values.min(axis=-1)
    if pool == "max":
        return values.max(axis=-1)
    if pool in ("mean", "avg", "average"):
        return values.mean(axis=-1)
    raise ValueError(f"Unknown pool {pool!r}; use min, max, or mean.")


def downsample_utterance_frames(
    frame_labels: np.ndarray,
    frame_scores: np.ndarray,
    factor: int,
    pool: str = "min",
    label_merge_rule: str = "any_spoof",
    ignore_index: int = -100,
    bonafide_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample one utterance from source hop to a coarser hop (integer factor).

    Labels are merged with ``label_merge_rule`` (``any_spoof``, ``all_spoof``,
    ``majority``, ``any_non_bonafide`` — same as dataset ``label_merge_rule``).
    Scores use ``pool`` (``min`` | ``max`` | ``mean``).
    """
    from deepfense.data.temporal_utils import downsample_frame_labels, merge_label_window

    valid = frame_labels != ignore_index
    fl = frame_labels[valid]
    fs = frame_scores[valid]
    if factor <= 1 or fl.size == 0:
        return fl, fs

    spoof_label = abs(1 - bonafide_label)
    wl = downsample_frame_labels(
        fl,
        factor,
        rule=label_merge_rule,
        spoof_label=spoof_label,
        bonafide_label=bonafide_label,
        ignore_value=ignore_index,
    )
    ws_parts = []
    n_full = fl.size // factor
    if n_full > 0:
        bs = fs[: n_full * factor].reshape(n_full, factor)
        ws_parts.append(_pool_axis(bs, pool))

    rem = fl.size - n_full * factor
    if rem > 0:
        tail_l = fl[n_full * factor :]
        tail_s = fs[n_full * factor :]
        tail_label = merge_label_window(
            tail_l,
            rule=label_merge_rule,
            spoof_label=spoof_label,
            bonafide_label=bonafide_label,
            ignore_value=ignore_index,
        )
        if tail_label != ignore_index:
            wl = np.concatenate([wl, np.array([tail_label], dtype=wl.dtype)])
        ws_parts.append(np.array([pool_utterance_scores(tail_s, pool=pool)], dtype=fs.dtype))

    if not ws_parts:
        return wl, np.array([], dtype=fs.dtype)
    return wl, np.concatenate(ws_parts)


def frames_at_resolution(
    keys: Sequence,
    labels: np.ndarray,
    scores: np.ndarray,
    source_ms: float,
    target_ms: float,
    pool: str = "min",
    label_merge_rule: str = "any_spoof",
    ignore_index: int = -100,
    bonafide_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool per-utterance frames to ``target_ms`` (must be multiple of ``source_ms``)."""
    if target_ms < source_ms:
        raise ValueError(
            f"target_ms ({target_ms}) must be >= source_ms ({source_ms}); "
            "upsampling labels is not supported."
        )
    factor = int(round(target_ms / source_ms))
    if abs(factor * source_ms - target_ms) > 1e-6:
        raise ValueError(
            f"target_ms ({target_ms}) must be an integer multiple of source_ms ({source_ms})."
        )

    utterances = group_frames_by_utterance(keys, labels, scores)
    out_l, out_s = [], []
    for frame_labels, frame_scores in utterances.values():
        dl, ds = downsample_utterance_frames(
            frame_labels,
            frame_scores,
            factor=factor,
            pool=pool,
            label_merge_rule=label_merge_rule,
            ignore_index=ignore_index,
            bonafide_label=bonafide_label,
        )
        if dl.size:
            out_l.append(dl)
            out_s.append(ds)
    if not out_l:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    return np.concatenate(out_l), np.concatenate(out_s)
