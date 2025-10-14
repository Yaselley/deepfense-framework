import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from deepfense.training.evaluations.registry import register_eval


@register_eval("F1")
def compute_f1(labels: np.ndarray, predictions: np.ndarray):
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    return macro_f1, per_class_f1


@register_eval("Accuracy")
def compute_accuracy(labels: np.ndarray, predictions: np.ndarray):
    acc = accuracy_score(labels, predictions)
    return acc
