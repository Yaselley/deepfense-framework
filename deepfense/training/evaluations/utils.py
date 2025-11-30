import numpy as np


def _metric_get_1d_scores(raw_scores: np.ndarray, metric_params: dict) -> np.ndarray:
    """
    Converts raw [N, C] model output into a 1D score array
    (where higher = bonafide) based on params.
    """
    # If scores are already 1D or [N, 1], squeeze them
    if raw_scores.ndim == 1:
        return raw_scores
    if raw_scores.shape[1] == 1:
        return raw_scores.squeeze(1)

    # Get loss_type from the metric's parameters (e.g., from your config file)
    loss_type = metric_params.get("loss", "crossentropy").lower()

    # Get the bonafide_label (defaults to 1)
    bonafide_label = metric_params.get("bonafide_label", 1)
    spoof_label = abs(1 - bonafide_label)

    if loss_type == "crossentropy":
        return raw_scores[:, bonafide_label]
    elif "ocsoftmax" in loss_type:
        # OC-Softmax usually returns single value (cos_theta) from get_logits
        # But if it somehow returned 2 values, handle it. 
        # Currently get_logits returns [B, 1] usually, caught above.
        # If [B, 2] (pos, neg), we take pos.
        return raw_scores[:, 0] 
    else:
        # AMSoftmax, ASoftmax, etc.
        return raw_scores[:, bonafide_label] - raw_scores[:, spoof_label]

