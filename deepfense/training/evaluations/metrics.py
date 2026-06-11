import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, jaccard_score
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


@register_metric("FRAME_ACC")
def compute_frame_accuracy(labels, scores, params):
    r = compute_accuracy(labels, scores, params)
    return {"FRAME_ACC": r["ACC"]}


@register_metric("FRAME_F1")
def compute_frame_f1(labels, scores, params):
    r = compute_f1(labels, scores, params)
    return {"FRAME_F1": r["F1_SCORE"]}


@register_metric("FRAME_AUC")
def compute_frame_auc(labels, scores, params):
    """ROC-AUC on frame-level scores (LLR ranking)."""
    try:
        if len(np.unique(labels)) < 2:
            return {"FRAME_AUC": float("nan")}
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return {"FRAME_AUC": auc}


@register_metric("FRAME_JACCARD_SPOOF")
def compute_frame_jaccard_spoof(labels, scores, params):
    """
    Jaccard index on binary *fake* detection (label 0 = spoof, 1 = bonafide):
    predictions are spoof where LLR < 0 (or argmax class for 2D scores).
    """
    spoof_label = int(params.get("spoof_label", 0))
    if scores.ndim == 2:
        pred_fake = (np.argmax(scores, axis=1) == spoof_label).astype(int)
    else:
        pred_fake = (scores < 0).astype(int)
    true_fake = (labels == spoof_label).astype(int)
    try:
        jac = jaccard_score(true_fake, pred_fake, average="binary", zero_division=0)
    except ValueError:
        jac = float("nan")
    return {"FRAME_JACCARD_SPOOF": jac}
