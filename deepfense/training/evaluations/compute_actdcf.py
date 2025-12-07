# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.utils import _metric_get_1d_scores


@register_metric("actDCF")
def compute_actDCF(labels, scores, params):
    """
    Compute the actual Detection Cost Function (actDCF).

    Args:
        labels (np.ndarray): Binary ground-truth labels (0 = bonafide, 1 = spoof)
        scores (np.ndarray): Raw [N, C] model prediction scores.
        params (dict):
            - Pspoof (float): Prior probability of spoof class
            - Cmiss (float): Cost of missing a bonafide sample
            - Cfa (float): Cost of falsely accepting a spoofed sample
            - bonafide_label (int, optional): Label representing bonafide (default: 0)
            - loss (str, optional): 'crossentropy' or 'amsoftmax'.
                                         (default: 'crossentropy')

    Returns:
        dict: {"actDCF": actDCF}
    """

    # Convert raw [N, C] scores to 1D based on the loss_type in params
    scores = _metric_get_1d_scores(scores, params)

    # ---- Validate input ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)  # scores is now 1D
    if labels.shape != scores.shape:
        print(labels.shape, scores.shape)
        raise ValueError("labels and scores must have the same shape")

    # ---- Extract parameters ----
    Pspoof = params.get("Pspoof", 0.5)
    Cmiss = params.get("Cmiss", 1.0)
    Cfa = params.get("Cfa", 1.0)
    bonafide_label = params.get("bonafide_label", 1)
    spoof_label = 1 - bonafide_label

    # ---- Compute threshold ----
    if Pspoof <= 0 or Pspoof >= 1:
        raise ValueError("Pspoof must be in (0, 1)")
    beta = (Cmiss * (1 - Pspoof)) / (Cfa * Pspoof)

    # The threshold is applied to the 1D scores
    threshold = -np.log(beta)

    # ---- Split scores ----
    bona_scores = scores[labels == bonafide_label]
    spoof_scores = scores[labels == spoof_label]

    if bona_scores.size == 0:
        rate_miss = 0.0
    else:
        rate_miss = np.mean(bona_scores < threshold)  # Miss: bona score < threshold

    if spoof_scores.size == 0:
        rate_fa = 0.0
    else:
        rate_fa = np.mean(
            spoof_scores >= threshold
        )  # False Alarm: spoof score >= threshold

    # ---- Compute normalized DCF ----
    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa
    denom = np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
    act_dcf /= denom if denom > 0 else np.nan

    return {"actDCF": act_dcf}
