import numpy as np

def _metric_get_1d_scores(raw_scores: np.ndarray, metric_params: dict) -> np.ndarray:
    """
    Converts raw [N, C] model output into a 1D score array
    (where higher = bonafide) based on params.
    """
    # If scores are already 1D, return them
    if raw_scores.ndim == 1:
        return raw_scores
    
    # If [N, 1], just squeeze (common for OC-Softmax or Sigmoid outputs)
    if raw_scores.shape[1] == 1:
        return raw_scores.squeeze(1)

    # Get loss_type from the metric's parameters (e.g., from your config file)
    loss_type = metric_params.get("loss", "crossentropy").lower()

    # Get the bonafide_label (defaults to 1)
    bonafide_label = metric_params.get("bonafide_label", 1)
    # Ensure spoof_label is the opposite of bonafide_label (assuming binary 0/1)
    spoof_label = abs(1 - bonafide_label)

    # --- Logic for different losses ---
    
    if "ocsoftmax" in loss_type:
        # OC-Softmax usually returns single value (cos_theta to bonafide center)
        # We've already handled [N, 1] above, but if it comes here as [N, 2] 
        # (unlikely given get_logits impl), we take column 0.
        return raw_scores[:, 0] 
    
    else:
        # For CrossEntropy (Softmax) and Margin-based Losses (AMSoftmax, ASoftmax):
        # These models output a score (logit or cosine similarity) for EACH class.
        #
        # Why take (Bonafide - Spoof)?
        # 1. This is the Log-Likelihood Ratio (LLR) for Softmax classifiers, which is 
        #    the optimal decision variable for binary classification.
        # 2. It is robust: A sample might have a "Bonafide Score" of 0.5, which looks 
        #    okay, but if its "Spoof Score" is 0.9, it is definitely a spoof.
        #    Using only 'Bonafide Score' ranks it as 0.5.
        #    Using 'Bonafide - Spoof' ranks it as -0.4 (correctly penalizing it).
        # 3. It produces the exact same ranking (and thus EER) as using the 
        #    Softmax Probability of the bonafide class, but is numerically more stable.
        return raw_scores[:, bonafide_label] - raw_scores[:, spoof_label]
