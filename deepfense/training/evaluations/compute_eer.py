

import numpy as np
import logging
from deepfense.utils.registry import register_metric
from deepfense.training.evaluations.utils import _metric_get_1d_scores

# Get a logger for this module
logger = logging.getLogger(__name__)


def compute_det_curve_legacy(labels, scores, bonafide_label=1):
    """
    Compute the DET curve values.
    Legacy version https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip
    Legacy version does not solve the tie when scores are equal 

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

def compute_det_curve(labels, scores, bonafide_label=1):
    """
    Compute the DET curve values.
    Compared with the legacy version, samples with the same score
    are now grouped together.

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
    # 
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    spoof_label = 1 - bonafide_label

    # bona/truth trials
    ts = scores[labels == bonafide_label]
    # spoof/fake trials
    ns = scores[labels == spoof_label]   

    if ts.size == 0 or ns.size == 0:
        raise ValueError("Both bonafide and spoof samples must be present")

        
    all_scores = np.unique(np.concatenate((ts, ns)))
    
    # count target and non-target samples for each score
    def count_ele(s):
        return [np.sum(ts == s), np.sum(ns == s)]

    cnt = np.array(list(map(count_ele, all_scores)))
    indices = np.argsort(all_scores, kind='mergesort')
    all_scores = all_scores[indices]
    
    # sorted based on score values
    t_labels = cnt[indices, 0]
    n_labels = cnt[indices, 1]

    # compute false rejection and false acceptance rates                                                       
    tar_trial_sums = np.cumsum(t_labels)
    nontarget_trial_sums = np.cumsum(n_labels)

    # false rejection rates                                                                                    
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / ts.size))
    # false acceptance rates
    far = 1-np.concatenate((np.atleast_1d(0), nontarget_trial_sums / ns.size))
    
    # Thresholds are the sorted scores                                                                         
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[0] - 0.001), all_scores+np.finfo(np.float32).eps))

    return frr, far, thresholds


def _det_curve_for_eer(labels, scores_1d, bonafide_label):
    """Pick DET implementation: legacy sort is O(n log n); grouped unique is slow on float LLRs."""
    if labels.size >= 50_000:
        return compute_det_curve_legacy(labels, scores_1d, bonafide_label)
    return compute_det_curve(labels, scores_1d, bonafide_label)


def _eer_from_labels_scores(labels, scores_1d, bonafide_label, params, precise=False):
    """Core EER computation on 1D scores and integer labels."""
    labels = np.asarray(labels).astype(int)
    scores_1d = np.asarray(scores_1d, dtype=float)
    bonafide_label = int(bonafide_label)

    frr, far, thresholds = _det_curve_for_eer(labels, scores_1d, bonafide_label)
    abs_diffs = np.abs(frr - far)
    min_index = int(np.argmin(abs_diffs))
    eer = float(np.mean((frr[min_index], far[min_index])))

    precise = params.get("precise", precise)
    if precise:
        return {
            "EER": eer,
            "EER_threshold": float(thresholds[min_index]),
            "FRR": frr.tolist(),
            "FAR": far.tolist(),
            "thresholds": thresholds.tolist(),
        }
    return {"EER": eer}


@register_metric("EER")
def compute_eer(labels, scores, params, precise=False):
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    Clip-level by default. For temporal / partial-spoof evaluation, set
    ``pool: min`` (or ``max`` / ``mean``) to derive utterance-level EER from
    framewise scores — same as PartialSpoof ``get_utteer_by_seg``.

    Args:
        labels (np.ndarray): Binary ground-truth labels
                             (1 = bonafide, 0 = spoof by default)
        scores (np.ndarray): Raw [N, C] model prediction scores.
        params (dict):
            - bonafide_label (int, optional): Label for bonafide class (default: 1)
            - loss (str, optional): 'crossentropy' or 'amsoftmax'.
            - pool (str, optional): min/max/mean — pool frames per utterance when
              ``keys`` are provided (PartialSpoof utterance EER from segments).
            - keys (array-like): utterance id per frame (from temporal eval)
            - ignore_index (int): padded frame label to skip when pooling (default -100)
            - precise (bool, optional): From config, to return full DET data.

    Returns:
        dict: { "EER": ... }
    """
    scores_1d = _metric_get_1d_scores(scores, params)
    bonafide_label = params.get("bonafide_label", 1)

    pool = params.get("pool")
    keys = params.get("keys")
    if pool and keys is not None and len(keys) == len(labels):
        from deepfense.training.evaluations.temporal_eer_utils import pool_frames_to_utterances

        labels, scores_1d = pool_frames_to_utterances(
            keys,
            np.asarray(labels).astype(int),
            scores_1d,
            bonafide_label=int(bonafide_label),
            pool=str(pool),
            ignore_index=int(params.get("ignore_index", -100)),
        )

    return _eer_from_labels_scores(labels, scores_1d, bonafide_label, params, precise)
