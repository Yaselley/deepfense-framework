# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

def compute_det_curve(labels, scores, bonafide_label=0):
    """
    Compute the DET curve values.

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (0 = bonafide, 1 = spoof by default)
        scores (np.ndarray): Model prediction scores (higher → more likely spoof)
        bonafide_label (int): Label representing bonafide class (default: 0)

    Returns:
        tuple: (frr, far, thresholds)
            - frr: np.ndarray, False Rejection Rates (#N,)
            - far: np.ndarray, False Acceptance Rates (#N,)
            - thresholds: np.ndarray, thresholds corresponding to FRR/FAR
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    spoof_label = 1 - bonafide_label

    target_scores = scores[labels == bonafide_label]   # bona/truth trials
    nontarget_scores = scores[labels == spoof_label]   # spoof/fake trials

    if target_scores.size == 0 or nontarget_scores.size == 0:
        raise ValueError("Both bonafide and spoof samples must be present")

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    all_labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels by ascending score (important for DET consistency)
    indices = np.argsort(all_scores, kind='mergesort')
    sorted_labels = all_labels[indices]

    # Cumulative sums
    tar_trial_sums = np.cumsum(sorted_labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # Compute FRR (miss rate) and FAR (false acceptance rate)
    frr = np.concatenate(([0.0], tar_trial_sums / target_scores.size))
    far = np.concatenate(([1.0], nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        ([all_scores[indices[0]] - 1e-6], all_scores[indices])
    )

    return frr, far, thresholds


@register_eval("EER")
def compute_eer(labels, scores, params):
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (0 = bonafide, 1 = spoof by default)
        scores (np.ndarray): Model prediction scores (higher → more likely spoof)
        params (dict):
            - bonafide_label (int, optional): Label representing bonafide class (default: 0)

    Returns:
        tuple: {
        "EER": float(eer),
        "EER_threshold": float(thresholds[min_index]),
        "FRR": frr.tolist(),
        "FAR": far.tolist(),
        "thresholds": thresholds.tolist(),
    }
    """
    bonafide_label = params.get("bonafide_label", 0)
    print(scores)
    frr, far, thresholds = compute_det_curve(labels, scores, bonafide_label)

    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))

    return {
        "EER": float(eer),
        "EER_threshold": float(thresholds[min_index]),
        "FRR": frr.tolist(),
        "FAR": far.tolist(),
        "thresholds": thresholds.tolist(),
    }