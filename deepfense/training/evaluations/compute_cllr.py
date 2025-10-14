# https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

import numpy as np
from deepfense.training.evaluations.registry import register_eval

@register_eval("CLLR")
def calculate_CLLR(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores.
    
    Parameters:
    target_llrs (list or numpy array): Log-likelihood ratios for target trials.
    nontarget_llrs (list or numpy array): Log-likelihood ratios for non-target trials.
    
    Returns:
    float: The calculated CLLR value.
    """
    def negative_log_sigmoid(lodds):
        """
        Calculate the negative log of the sigmoid function.
        
        Parameters:
        lodds (numpy array): Log-odds values.
        
        Returns:
        numpy array: The negative log of the sigmoid values.
        """
        return np.log1p(np.exp(-lodds))

    # Convert the input lists to numpy arrays if they are not already
    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)
    
    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(target_llrs)) + np.mean(negative_log_sigmoid(-nontarget_llrs))) / np.log(2)
    
    return cllr
