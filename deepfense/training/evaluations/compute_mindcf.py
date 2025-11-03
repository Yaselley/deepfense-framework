# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

@register_eval("minDCF")
def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    # prior of target class
    p_target = 1- Pspoof
    
    # detection cost at all operation points
    c_det = Cmiss * frr  * p_target + Cfa * far * (1 - p_target)

    # find the minimium operation point
    mindcf_idx = np.argmin(c_det)
    min_c_det = c_det[mindcf_idx]
    min_c_det_threshold = thresholds[mindcf_idx]

    # Normalize the cost.
    #  floor_c_dct: a dummy system that accept or reject all
    floor_c_dcf = min(Cmiss * p_target, Cfa * (1 - p_target))
    
    min_dcf = min_c_det / floor_c_dcf
    return min_dcf, min_c_det_threshold
