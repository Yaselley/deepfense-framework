"""Range-based Equal Error Rate for framewise spoof localization."""

import logging

import numpy as np

from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.temporal_eer_utils import (
    prepare_temporal_scores,
    resolve_range_hop_sec,
    search_range_eer,
)

logger = logging.getLogger(__name__)


@register_metric("RANGE_EER")
def compute_range_eer(labels, scores, params):
    """
    Range-EER for partial-spoof / localization evaluation.

    Requires flattened frame labels/scores and per-frame ``keys`` from temporal
    eval. Threshold is searched so duration-overlap false-alarm and miss rates
    are equal (Zhang et al., INTERSPEECH 2023; PartialSpoof ``RangeEER.py``).

    Params:
        keys: utterance id per frame (required)
        label_hop_ms / source_ms: native prediction hop in ms
        source_label_hop_ms / reference_ms: reference hop for overlap (default: prediction hop)
        bonafide_label (int): bonafide class index (default 1)
        ignore_index (int): padded/invalid frame label (default -100)
        prec (float): bisection stop tolerance on |FPR-FNR| (default 1e-4)
        max_iters (int): max bisection steps (default 64)
    """
    keys = params.get("keys")
    if keys is None or len(keys) == 0:
        logger.warning("RANGE_EER requires per-frame keys from temporal eval.")
        return {"RANGE_EER": float("nan")}

    if len(keys) != len(labels):
        logger.warning("RANGE_EER: keys length != labels length.")
        return {"RANGE_EER": float("nan")}

    labels, scores = prepare_temporal_scores(labels, scores, params)

    bonafide_label = int(params.get("bonafide_label", 1))
    spoof_label = abs(1 - bonafide_label)
    ignore_index = int(params.get("ignore_index", -100))
    prec = float(params.get("prec", 1e-4))
    max_iters = int(params.get("max_iters", 64))
    hop_sec = resolve_range_hop_sec(params)

    valid = labels != ignore_index
    vl = labels[valid]
    if vl.size == 0 or len(np.unique(vl)) < 2:
        logger.warning("RANGE_EER: not enough valid frame labels.")
        return {"RANGE_EER": float("nan")}

    range_eer, threshold = search_range_eer(
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

    ref_ms = int(round(hop_sec * 1000.0))
    out = {"RANGE_EER": range_eer, f"RANGE_EER_{ref_ms}ms": range_eer}
    if params.get("precise", False):
        out["RANGE_EER_threshold"] = threshold
    return out
