# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

@register_eval("CLLR")
def calculate_CLLR(labels, scores, params):
    """
    Compute the log-likelihood ratio cost (CLLR).

    Args:
        labels (np.ndarray): Binary ground-truth labels 
                             (0 = bonafide, 1 = spoof by default)
        scores (np.ndarray): Log-likelihood ratio scores (LLRs)
                             (higher → more likely spoof)
        params (dict):
            - bonafide_label (int, optional): Label representing bonafide class (default: 0)

    Returns:
        {"CLLR": cllr}
    """

    # ---- Validate input ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if labels.shape != scores.shape:
        print(labels.shape, scores.shape)
        raise ValueError("labels and scores must have the same shape")

    bonafide_label = params.get("bonafide_label", 0)
    spoof_label = 1 - bonafide_label

    # ---- Split scores ----
    bona_scores = scores[labels == bonafide_label]
    spoof_scores = scores[labels == spoof_label]

    if bona_scores.size == 0 or spoof_scores.size == 0:
        raise ValueError("Both bonafide and spoof samples must be present")

    # ---- Helper: negative log sigmoid ----
    def negative_log_sigmoid(x):
        # log(1 + exp(-x)) — numerically stable
        return np.log1p(np.exp(-x))

    # ---- Compute CLLR ----
    term1 = np.mean(negative_log_sigmoid(bona_scores))
    term2 = np.mean(negative_log_sigmoid(-spoof_scores))
    cllr = 0.5 * (term1 + term2) / np.log(2)  # log base 2 normalization

    return {"CLLR": cllr}
