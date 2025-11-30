# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.compute_eer import compute_eer
from deepfense.training.evaluations.utils import _metric_get_1d_scores


@register_metric("minDCF")
def compute_mindcf(labels, scores, params):
    """
    Compute the minimum normalized Detection Cost Function (minDCF).

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (1 = bonafide, 0 = spoof by default)
        scores (np.ndarray): Raw [N, C] model prediction scores.
        params (dict):
            - Pspoof (float): Prior probability of spoof class (default: 0.5)
            - Cmiss (float): Cost of missing a bonafide sample (default: 1.0)
            - Cfa (float): Cost of falsely accepting a spoof sample (default: 1.0)
            - bonafide_label (int, optional): Label representing bonafide class (default: 1)
            - loss (str, optional): 'crossentropy' or 'amsoftmax'.
                                         (default: 'crossentropy')

    Returns:
        {
        "minDCF": min_dcf,
        "minDCF_threshold": min_c_det_threshold
    }
    """

    # --- 1. ADD THIS LINE ---
    # Convert raw [N, C] scores to 1D based on the loss_type in params
    scores = _metric_get_1d_scores(scores, params)
    # --- END OF CHANGE ---

    # ---- Validate inputs ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)  # scores is now 1D
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    # ---- Extract parameters ----
    Pspoof = params.get("Pspoof", 0.5)
    Cmiss = params.get("Cmiss", 1.0)
    Cfa = params.get("Cfa", 1.0)
    # bonafide_label is handled by compute_eer, but we can get it here for clarity
    bonafide_label = params.get("bonafide_label", 1)

    # ---- Compute DET curve ----
    # 'scores' is now 1D, and we pass 'params' so compute_eer
    # also gets bonafide_label, loss_type, etc.
    eer_metrics = compute_eer(labels, scores, params, precise=True)

    frr = np.array(eer_metrics.get("FRR"))
    far = np.array(eer_metrics.get("FAR"))
    thresholds = np.array(eer_metrics.get("thresholds"))

    # Handle case where compute_eer failed (e.g., missing samples)
    if frr.size <= 1 or far.size <= 1:
        import logging

        logging.warning("minDCF calculation failed: invalid DET curve data.")
        return {"minDCF": np.nan, "minDCF_threshold": np.nan}

    Pbonafide = 1 - Pspoof
    min_c_det = float("inf")
    min_c_det_threshold = None

    # ---- Find minimum detection cost ----
    for i in range(len(frr)):
        c_det = Cmiss * frr[i] * Pbonafide + Cfa * far[i] * Pspoof
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # ---- Normalize DCF ----
    denom = np.min([Cmiss * Pbonafide, Cfa * Pspoof])
    min_dcf = min_c_det / denom if denom > 0 else np.nan

    return {"minDCF": float(min_dcf), "minDCF_threshold": float(min_c_det_threshold)}
