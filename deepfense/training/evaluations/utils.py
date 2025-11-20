import numpy as np


def _metric_get_1d_scores(raw_scores: np.ndarray, metric_params: dict) -> np.ndarray:
    """
    Converts raw [N, C] model output into a 1D score array
    (where higher = spoof) based on params.
    """
    # If scores are already 1D, do nothing
    if raw_scores.ndim == 1:
        return raw_scores

    # Get loss_type from the metric's parameters (e.g., from your config file)
    loss_type = metric_params.get("loss", "crossentropy").lower()

    # Get the bonafide_label (defaults to 1)
    bonafide_label = metric_params.get("bonafide_label", 1)
    spoof_label = abs(1 - bonafide_label)

    if loss_type != "crossentropy":
        return raw_scores[:, bonafide_label] - raw_scores[:, spoof_label]

    else:
        return raw_scores[:, bonafide_label]
