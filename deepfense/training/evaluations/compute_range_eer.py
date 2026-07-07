"""Range-based Equal Error Rate for framewise spoof localization."""

import logging

import numpy as np

from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.temporal_eer_utils import (
    prepare_temporal_scores,
    search_range_eer,
)

logger = logging.getLogger(__name__)


@register_metric("RANGE_EER")
def compute_range_eer(labels, scores, params):
    """
    Range-EER for partial-spoof / localization evaluation.

    Requires flattened frame labels/scores from temporal evaluation. Threshold
    is searched so frame-level false-alarm and miss rates are equal (Range-EER).

    Params:
        bonafide_label (int): bonafide class index (default 1)
        ignore_index (int): padded/invalid frame label (default -100)
        prec (float): bisection stop tolerance on |FPR-FNR| (default 1e-4)
        max_iters (int): max bisection steps (default 64)
    """
    labels, scores = prepare_temporal_scores(labels, scores, params)

    bonafide_label = int(params.get("bonafide_label", 1))
    spoof_label = abs(1 - bonafide_label)
    ignore_index = int(params.get("ignore_index", -100))
    prec = float(params.get("prec", 1e-4))
    max_iters = int(params.get("max_iters", 64))

    range_eer, threshold = search_range_eer(
        labels=labels,
        scores=scores,
        spoof_label=spoof_label,
        bonafide_label=bonafide_label,
        ignore_index=ignore_index,
        prec=prec,
        max_iters=max_iters,
    )

    if params.get("precise", False):
        return {
            "RANGE_EER": range_eer,
            "RANGE_EER_threshold": threshold,
        }
    return {"RANGE_EER": range_eer}
