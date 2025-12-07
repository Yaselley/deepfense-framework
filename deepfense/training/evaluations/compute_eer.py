# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
import logging
from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.utils import _metric_get_1d_scores

# Get a logger for this module
logger = logging.getLogger(__name__)


def compute_det_curve(labels, scores, bonafide_label=1):
    """
    Compute the DET curve values.

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (1 = bonafide, 0 = spoof by default)
        scores (np.ndarray): 1D model prediction scores (higher → more likely spoof)
        bonafide_label (int): Label representing bonafide class (default: 0)

    Returns:
        tuple: (frr, far, thresholds)
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    spoof_label = 1 - bonafide_label

    target_scores = scores[labels == bonafide_label]  # bona/truth trials
    nontarget_scores = scores[labels == spoof_label]  # spoof/fake trials

    if target_scores.size == 0 or nontarget_scores.size == 0:
        # --- MODIFICATION: Log a warning instead of crashing ---
        logger.warning(
            "DET curve calculation failed: missing bonafide or spoof samples."
        )
        # Return sensible defaults to avoid crashing EER calculation
        return np.array([0.0]), np.array([1.0]), np.array([0.0])
        # --- END MODIFICATION ---

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    all_labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels by ascending score (important for DET consistency)
    indices = np.argsort(all_scores, kind="mergesort")
    sorted_labels = all_labels[indices]

    # Cumulative sums
    tar_trial_sums = np.cumsum(sorted_labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # Compute FRR (miss rate) and FAR (false acceptance rate)
    frr = np.concatenate(([0.0], tar_trial_sums / target_scores.size))
    far = np.concatenate(([1.0], nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(([all_scores[indices[0]] - 1e-6], all_scores[indices]))

    return frr, far, thresholds


@register_metric("EER")
def compute_eer(labels, scores, params, precise=False):
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (1 = bonafide, 0 = spoof by default)
        scores (np.ndarray): Raw [N, C] model prediction scores.
        params (dict):
            - bonafide_label (int, optional): Label for bonafide class (default: 0)
            - loss (str, optional): 'crossentropy' or 'amsoftmax'.
            - precise (bool, optional): From config, to return full DET data.

    Returns:
        dict: { "EER": ... }
    """

    # Convert raw [N, C] scores to 1D based on the loss_type in params
    scores_1d = _metric_get_1d_scores(scores, params)

    bonafide_label = params.get("bonafide_label", 1)

    frr, far, thresholds = compute_det_curve(labels, scores_1d, bonafide_label)

    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))

    precise = params.get("precise", precise)

    if precise:
        return {
            "EER": float(eer),
            "EER_threshold": float(thresholds[min_index]),
            "FRR": frr.tolist(),
            "FAR": far.tolist(),
            "thresholds": thresholds.tolist(),
        }
    else:
        return {
            "EER": float(eer),
        }
