# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

@register_eval("actDCF")
def compute_actDCF(bonafide_scores, spoof_scores, Pspoof, Cmiss, Cfa):
    """
    compute actual DCF, given threshold decided by prior and decision costs

    input
    -----
      bonafide_scores: np.array, scores of bonafide data
      spoof_scores: np.array, scores of spoof data
      Pspoof: scalar, prior probabiltiy of spoofed class
      Cmiss: scalar, decision cost of missing a bonafide sample
      Cfa: scalar, decision cost of falsely accept a spoofed sample

    output
    ------
      actDCF: scalar, actual DCF normalized
      threshold: scalar, threshold for making the decision
    """
    # the beta in evaluation plan (eq.(3))
    beta = Cmiss * (1 - Pspoof) / (Cfa * Pspoof)
    
    # compute the decision threshold based on
    threshold = - np.log(beta)

    # miss rate
    rate_miss = np.sum(bonafide_scores < threshold) / bonafide_scores.size

    # fa rate
    rate_fa = np.sum(spoof_scores >= threshold) / spoof_scores.size

    # unnormalized DCF
    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa

    # normalized DCF
    act_dcf = act_dcf / np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
    
    return act_dcf, threshold
    