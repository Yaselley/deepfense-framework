# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval
from deepfense.training.evaluations.compute_eer import compute_eer

@register_eval("minDCF")
def compute_mindcf(labels, scores, params):
    """
    Compute the minimum normalized Detection Cost Function (minDCF).

    Args:
        labels (np.ndarray): Binary ground-truth labels 
                             (0 = bonafide, 1 = spoof by default)
        scores (np.ndarray): Model prediction scores (higher → more likely spoof)
        params (dict):
            - Pspoof (float): Prior probability of spoof class (default: 0.5)
            - Cmiss (float): Cost of missing a bonafide sample (default: 1.0)
            - Cfa (float): Cost of falsely accepting a spoof sample (default: 1.0)
            - bonafide_label (int, optional): Label representing bonafide class (default: 0)

    Returns:
        {
        "minDCF": min_dcf,
        "minDCF_threshold": min_c_det_threshold
    }

    """

    import numpy as np

    # ---- Validate inputs ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    # ---- Extract parameters ----
    Pspoof = params.get("Pspoof", 0.5)
    Cmiss = params.get("Cmiss", 1.0)
    Cfa = params.get("Cfa", 1.0)
    bonafide_label = params.get("bonafide_label", 0)

    # ---- Compute DET curve ----
    eer_metrics = compute_eer(labels, scores, params)
    frr = eer_metrics["FRR"]
    far = eer_metrics["FAR"]
    thresholds = eer_metrics["thresholds"]

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

    return {
        "minDCF": min_dcf,
        "minDCF_threshold": min_c_det_threshold
    }
