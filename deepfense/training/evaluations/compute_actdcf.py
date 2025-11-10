# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

import numpy as np

@register_eval("actDCF")
def compute_actDCF(labels, scores, params):
    """
    Compute the actual Detection Cost Function (actDCF).

    Args:
        labels (np.ndarray): Binary ground-truth labels (0 = bonafide, 1 = spoof)
        scores (np.ndarray): Model prediction scores (higher → more likely spoof)
        params (dict):
            - Pspoof (float): Prior probability of spoof class
            - Cmiss (float): Cost of missing a bonafide sample
            - Cfa (float): Cost of falsely accepting a spoofed sample
            - bonafide_label (int, optional): Label representing bonafide (default: 0)

    Returns:
        dicr: {"actDCF": actDCF}
    """

    # ---- Validate input ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.shape != scores.shape:
        print(labels.shape, scores.shape)
        raise ValueError("labels and scores must have the same shape")

    # ---- Extract parameters ----
    Pspoof = params.get("Pspoof", 0.5)
    Cmiss = params.get("Cmiss", 1.0)
    Cfa = params.get("Cfa", 1.0)
    bonafide_label = params.get("bonafide_label", 0)
    spoof_label = 1 - bonafide_label

    # ---- Compute threshold ----
    if Pspoof <= 0 or Pspoof >= 1:
        raise ValueError("Pspoof must be in (0, 1)")
    beta = (Cmiss * (1 - Pspoof)) / (Cfa * Pspoof)
    threshold = -np.log(beta)

    # ---- Split scores ----
    bona_scores = scores[labels == bonafide_label]
    spoof_scores = scores[labels == spoof_label]

    # ---- Compute rates ----
    rate_miss = np.mean(bona_scores < threshold)
    rate_fa = np.mean(spoof_scores >= threshold)

    # ---- Compute normalized DCF ----
    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa
    denom = np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
    act_dcf /= denom if denom > 0 else np.nan

    return {"actDCF": act_dcf}
