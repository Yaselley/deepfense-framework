# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.utils import _metric_get_1d_scores


@register_metric("CLLR")
def calculate_CLLR(labels, scores, params):
    """
    Compute the log-likelihood ratio cost (CLLR).

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (1 = bonafide, 0 = spoof by default)
        scores (np.ndarray): Raw [N, C] model output scores (LLRs).
        params (dict):
            - bonafide_label (int, optional): Label for bonafide class (default: 1)
            - loss (str, optional): 'crossentropy' or 'amsoftmax'.
                                         (default: 'crossentropy')

    Returns:
        {"CLLR": cllr}
    """

    # --- 1. ADD THIS LINE ---
    # Convert raw [N, C] scores to 1D based on the loss_type in params
    scores = _metric_get_1d_scores(scores, params)
    # --- END OF CHANGE ---

    # ---- Validate input ----
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)  # scores is now 1D

    if labels.shape != scores.shape:
        print(labels.shape, scores.shape)
        raise ValueError("labels and scores must have the same shape")

    bonafide_label = params.get("bonafide_label", 1)
    spoof_label = 1 - bonafide_label

    # ---- Split scores ----
    bona_scores = scores[labels == bonafide_label]
    spoof_scores = scores[labels == spoof_label]

    if bona_scores.size == 0 or spoof_scores.size == 0:
        # Don't raise an error, just return NaN or a high value if no samples
        # Or log a warning. Let's log and return nan.
        import logging

        logger = logging.getLogger("CLLR")
        logger.warning("CLLR calculation failed: missing bonafide or spoof samples.")
        return {"CLLR": np.nan}

    # ---- Helper: negative log sigmoid ----
    def negative_log_sigmoid(x):
        # log(1 + exp(-x)) — numerically stable
        return np.log1p(np.exp(-x))

    # ---- Compute CLLR ----
    term1 = np.mean(negative_log_sigmoid(bona_scores))
    term2 = np.mean(negative_log_sigmoid(-spoof_scores))

    # log base 2 normalization
    cllr = (term1 + term2) * 0.5 / np.log(2)

    return {"CLLR": cllr}
