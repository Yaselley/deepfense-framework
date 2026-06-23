"""Multi-resolution SEGMENT / RANGE / utterance EER (PartialSpoof SegmentEER-style)."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.compute_eer import _eer_from_labels_scores
from deepfense.training.evaluations.temporal_eer_utils import (
    frames_at_resolution,
    group_frames_by_utterance,
    infer_utterance_label,
    normalize_resolutions_ms,
    pool_utterance_scores,
    prepare_temporal_scores,
    resolve_range_hop_sec,
    search_range_eer,
)

logger = logging.getLogger(__name__)

_DEFAULT_TYPES = ("segment", "range", "utterance")


def _resolve_resolutions_ms(params: dict) -> List[float]:
    resolutions = normalize_resolutions_ms(params.get("resolutions_ms"))
    if resolutions:
        return resolutions
    source = params.get("label_hop_ms") or params.get("source_ms")
    if source is not None:
        return [float(source)]
    return [20.0]


def _resolve_types(params: dict) -> tuple:
    raw = params.get("types", _DEFAULT_TYPES)
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(t).lower() for t in raw)


def _concat_pct(values: List[float]) -> str:
    return ",".join(f"{v * 100:.4f}" for v in values)


def _resolve_reference_ms(params: dict, source_ms: float) -> float:
    pred_ms = float(params.get("source_ms") or params.get("label_hop_ms") or source_ms)
    ref_ms = float(
        params.get("source_label_hop_ms")
        or params.get("reference_ms")
        or pred_ms
    )
    if ref_ms < pred_ms:
        logger.debug(
            "MULTIRES_EER: reference hop %.1f ms finer than eval frames %.1f ms; "
            "using eval hop for Range-EER.",
            ref_ms,
            pred_ms,
        )
        ref_ms = pred_ms
    return ref_ms


def _compute_utterance_eer(
    keys,
    labels: np.ndarray,
    scores: np.ndarray,
    params: dict,
    bonafide_label: int,
    ignore_index: int,
    pool: str,
) -> float:
    """Single utterance-level EER from native frame scores (PartialSpoof ``get_utteer_by_seg``)."""
    utterances = group_frames_by_utterance(keys, labels, scores)
    utt_l, utt_s = [], []
    for frame_labels, frame_scores in utterances.values():
        valid = frame_labels != ignore_index
        if not np.any(valid):
            continue
        fl = frame_labels[valid]
        fs = frame_scores[valid]
        utt_l.append(infer_utterance_label(fl, bonafide_label, ignore_index))
        utt_s.append(pool_utterance_scores(fs, pool=pool))
    if len(utt_l) < 2 or len(np.unique(utt_l)) < 2:
        return float("nan")
    return _eer_from_labels_scores(
        np.asarray(utt_l, dtype=int),
        np.asarray(utt_s, dtype=float),
        bonafide_label,
        params,
    )["EER"]


def _compute_range_eer(
    keys,
    labels: np.ndarray,
    scores: np.ndarray,
    params: dict,
    bonafide_label: int,
    spoof_label: int,
    ignore_index: int,
    prec: float,
    max_iters: int,
) -> float:
    """Single range-based EER at the finest available reference hop."""
    valid = labels != ignore_index
    vl = labels[valid]
    if vl.size == 0 or len(np.unique(vl)) < 2:
        return float("nan")
    hop_sec = resolve_range_hop_sec(params)
    rng, _ = search_range_eer(
        labels=labels,
        scores=scores,
        keys=keys,
        hop_sec=hop_sec,
        spoof_label=spoof_label,
        bonafide_label=bonafide_label,
        ignore_index=ignore_index,
        prec=prec,
        max_iters=max_iters,
    )
    return rng


@register_metric("MULTIRES_EER")
def compute_multires_eer(labels, scores, params):
    """
    EER at one or many frame resolutions (ms), with user-chosen score pooling.

    Downsamples native frame scores to each target hop (20 → 40 → 80 … ms),
    matching PartialSpoof SegmentEER upper-diagonal behaviour. Native
    resolution uses factor 1 (no pooling).

    Params:
        resolutions_ms: ``20`` or ``[20, 40, 80, 160]`` (default: ``label_hop_ms``)
        pool: ``min`` | ``max`` | ``mean`` for downsampling scores and
              utterance aggregation (default ``min``)
        label_merge_rule: ``any_spoof`` | ``all_spoof`` | ``majority`` |
              ``any_non_bonafide`` for merging labels when downsampling
              (default ``any_spoof``; inherits from ``data.label_merge_rule``)
        types: subset of ``segment``, ``range``, ``utterance`` (default: all).
              Utterance EER and Range-EER are each a **single** scalar from
              native eval frames (PartialSpoof ``get_utteer_by_seg`` /
              ``RangeEER.py``), not per downsampled resolution.
        bonafide_label: default 1
        ignore_index: default -100
        source_ms / label_hop_ms: native prediction hop in ms
        source_label_hop_ms / reference_ms: reference hop for Range-EER
        prec / max_iters: RANGE-EER bisection knobs

    Returns per-resolution ``SEGMENT_EER_*ms`` keys, a single ``RANGE_EER``
    (plus ``RANGE_EER_{ref}ms`` alias), and concatenated segment summaries.
    """
    keys = params.get("keys")
    if keys is None or len(keys) == 0:
        logger.warning("MULTIRES_EER requires per-frame keys from temporal eval.")
        return {"MULTIRES_EER": float("nan")}

    if len(keys) != len(labels):
        logger.warning("MULTIRES_EER: keys length != labels length.")
        return {"MULTIRES_EER": float("nan")}

    labels, scores = prepare_temporal_scores(labels, scores, params)
    bonafide_label = int(params.get("bonafide_label", 1))
    spoof_label = abs(1 - bonafide_label)
    ignore_index = int(params.get("ignore_index", -100))
    pool = str(params.get("pool", "min")).lower()
    label_merge_rule = str(params.get("label_merge_rule", "any_spoof"))
    source_ms = float(params.get("source_ms") or params.get("label_hop_ms") or 20.0)
    prec = float(params.get("prec", 1e-4))
    max_iters = int(params.get("max_iters", 64))
    types = _resolve_types(params)
    resolutions = _resolve_resolutions_ms(params)

    results = {}
    res_ms_str = []
    segment_vals = []

    if "utterance" in types:
        utt_eer = _compute_utterance_eer(
            keys, labels, scores, params, bonafide_label, ignore_index, pool
        )
        results["UTTERANCE_EER"] = utt_eer
        results["EER"] = utt_eer  # alias for legacy configs that used EER + pool: min

    if "range" in types:
        ref_ms = _resolve_reference_ms(params, source_ms)
        ref_key = int(round(ref_ms))
        range_eer = _compute_range_eer(
            keys,
            labels,
            scores,
            params,
            bonafide_label,
            spoof_label,
            ignore_index,
            prec,
            max_iters,
        )
        results["RANGE_EER"] = range_eer
        results[f"RANGE_EER_{ref_key}ms"] = range_eer

    for target_ms in resolutions:
        ms_key = int(round(target_ms))
        res_ms_str.append(str(ms_key))
        try:
            dl, ds = frames_at_resolution(
                keys,
                labels,
                scores,
                source_ms=source_ms,
                target_ms=target_ms,
                pool=pool,
                label_merge_rule=label_merge_rule,
                ignore_index=ignore_index,
                bonafide_label=bonafide_label,
            )
        except ValueError as exc:
            logger.warning("MULTIRES_EER skipped %sms: %s", ms_key, exc)
            if "segment" in types:
                results[f"SEGMENT_EER_{ms_key}ms"] = float("nan")
            continue

        if dl.size == 0 or len(np.unique(dl)) < 2:
            if "segment" in types:
                results[f"SEGMENT_EER_{ms_key}ms"] = float("nan")
            continue

        if "segment" in types:
            seg = _eer_from_labels_scores(dl, ds, bonafide_label, params)["EER"]
            results[f"SEGMENT_EER_{ms_key}ms"] = seg
            segment_vals.append(seg)

    results["RESOLUTIONS_ms"] = ",".join(res_ms_str)
    results["POOL"] = pool
    results["LABEL_MERGE_RULE"] = label_merge_rule
    if segment_vals:
        results["SEGMENT_EER_CONCAT_pct"] = _concat_pct(segment_vals)

    return results
