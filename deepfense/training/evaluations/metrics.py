import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from deepfense.training.evaluations.registry import register_eval


@register_eval("F1_SCORE")
def compute_f1(labels, predictions, params):
    macro_f1 = f1_score(labels, predictions, average=params.get("f1_average", "macro"), zero_division=0)
    return {"F1_SCORE": macro_f1}


@register_eval("ACC")
def compute_accuracy(labels, predictions):
    acc = accuracy_score(labels, predictions)
    return {"ACC": acc}
