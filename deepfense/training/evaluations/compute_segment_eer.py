"""Point-based segment-level EER (PartialSpoof SegmentEER, diagonal / same resolution)."""

import logging

import numpy as np

from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.compute_eer import _eer_from_labels_scores
from deepfense.training.evaluations.temporal_eer_utils import prepare_temporal_scores

logger = logging.getLogger(__name__)


@register_metric("SEGMENT_EER")
def compute_segment_eer(labels, scores, params):
    """
    Point-based segment EER: one trial per frame, standard DET on all frames.

    Matches PartialSpoof ``SegmentEER.py`` diagonal (DIGEER) at the training
    frame resolution. Requires flattened frame labels/scores from temporal eval.

    Params:
        ignore_index (int): drop padded frames (default -100)
        bonafide_label (int): bonafide class index (default 1)
    """
    labels, scores = prepare_temporal_scores(labels, scores, params)
    ignore_index = int(params.get("ignore_index", -100))
    valid = labels != ignore_index
    labels = labels[valid]
    scores = scores[valid]

    if labels.size == 0 or len(np.unique(labels)) < 2:
        logger.warning("SEGMENT_EER: not enough valid frame labels.")
        return {"SEGMENT_EER": float("nan")}

    bonafide_label = int(params.get("bonafide_label", 1))
    result = _eer_from_labels_scores(labels, scores, bonafide_label, params)
    return {"SEGMENT_EER": result["EER"]}
