# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

@register_eval("minDCF")
def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1- Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
