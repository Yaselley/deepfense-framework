import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from deepfense.utils.registry import register_metric


@register_metric("F1_SCORE")
def compute_f1(labels, scores, params):
    """
    Computes F1-score from raw scores.
    Handles 1D (binary) or 2D [N, C] (multi-class) scores.
    """
    if scores.ndim == 2:
        predictions = np.argmax(scores, axis=1)
    else:
        # Binary: 1D scores -> threshold at 0
        # This matches the trainer's `scores[:, 1] - scores[:, 0]` logic
        # (score > 0 means class 1)
        predictions = (scores > 0).astype(int)

    macro_f1 = f1_score(
        labels, predictions, average=params.get("f1_average", "macro"), zero_division=0
    )
    return {"F1_SCORE": macro_f1}


@register_metric("ACC")
def compute_accuracy(labels, scores, params):
    """
    Computes Accuracy from raw scores.
    Handles 1D (binary) or 2D [N, C] (multi-class) scores.

    (params is unused but kept for consistent signature)
    """
    if scores.ndim == 2:
        # Multi-class: [N, C] scores -> argmax
        predictions = np.argmax(scores, axis=1)
    else:
        # Binary: 1D scores -> threshold at 0
        predictions = (scores > 0).astype(int)

    acc = accuracy_score(labels, predictions)
    return {"ACC": acc}
