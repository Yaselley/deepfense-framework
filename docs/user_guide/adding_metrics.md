# Adding a New Metric

This guide shows you how to add a custom evaluation metric to DeepFense.

## Overview

Metrics in DeepFense compute evaluation scores from predictions and labels during validation and testing. They are registered with `@register_metric` and used by the evaluator to compute performance metrics. Metrics must handle both binary classification (1D scores) and multi-class (2D scores) scenarios.

## Step-by-Step Guide

### Step 1: Add to Metrics File

Add your metric function to `deepfense/training/evaluations/metrics.py`:

```python
from deepfense.utils.registry import register_metric
import numpy as np
from sklearn.metrics import precision_score, recall_score


@register_metric("Precision")
def compute_precision(labels, scores, params):
    """
    Compute Precision metric.
    
    Args:
        labels: Ground truth labels [N] (numpy array)
        scores: Predicted scores [N] or [N, C] (numpy array)
        params: Dictionary with metric parameters
            - threshold: Classification threshold (default: 0.0)
            - average: Averaging method for multi-class (default: 'macro')
    
    Returns:
        Dictionary with metric name and value: {"Precision": float}
    """
    threshold = params.get("threshold", 0.0)
    average = params.get("average", "macro")
    
    # Handle different score shapes
    if scores.ndim == 2:
        # Multi-class: [N, C] -> argmax
        predictions = np.argmax(scores, axis=1)
    else:
        # Binary: [N] -> threshold at 0
        predictions = (scores > threshold).astype(int)
    
    # Compute precision
    precision = precision_score(
        labels,
        predictions,
        average=average,
        zero_division=0
    )
    
    return {"Precision": precision}
```

### Step 2: Verify Registration

The metric is automatically registered when the module is imported. Check that it's registered:

```bash
deepfense list --component-type augmentations  # Note: metrics might be under augmentations or separate
```

Or programmatically:

```python
from deepfense.training.evaluations import metrics  # Import to register
from deepfense.utils.registry import METRIC_REGISTRY

# Check if registered
if "Precision" in METRIC_REGISTRY:
    print("Metric registered successfully!")
    print("Available metrics:", METRIC_REGISTRY.list())
```

### Step 3: Use in Configuration

Use your metric in a YAML configuration file:

```yaml
training:
  metrics:
    Precision:
      threshold: 0.5
      average: "macro"
    Recall: {}  # Using default parameters
    F1_SCORE: {}
    EER: {}
```

## Complete Example: Recall Metric

Here's a complete example for Recall:

```python
from deepfense.utils.registry import register_metric
import numpy as np
from sklearn.metrics import recall_score


@register_metric("Recall")
def compute_recall(labels, scores, params):
    """
    Compute Recall metric.
    """
    threshold = params.get("threshold", 0.0)
    average = params.get("average", "macro")
    
    # Convert scores to predictions
    if scores.ndim == 2:
        predictions = np.argmax(scores, axis=1)
    else:
        predictions = (scores > threshold).astype(int)
    
    recall = recall_score(
        labels,
        predictions,
        average=average,
        zero_division=0
    )
    
    return {"Recall": recall}
```

## Example: Area Under ROC Curve (AUC)

```python
from deepfense.utils.registry import register_metric
import numpy as np
from sklearn.metrics import roc_auc_score


@register_metric("AUC")
def compute_auc(labels, scores, params):
    """
    Compute Area Under ROC Curve.
    """
    # Handle multi-class scores
    if scores.ndim == 2:
        if scores.shape[1] == 2:
            # Binary classification: use positive class scores
            scores_1d = scores[:, 1]
        else:
            # Multi-class: use one-vs-rest
            try:
                auc = roc_auc_score(labels, scores, multi_class='ovr', average='macro')
                return {"AUC": auc}
            except:
                return {"AUC": 0.0}
    else:
        # Binary: 1D scores
        scores_1d = scores
    
    try:
        auc = roc_auc_score(labels, scores_1d)
        return {"AUC": auc}
    except ValueError:
        # Handle edge cases (e.g., all labels are the same)
        return {"AUC": 0.0}
```

## Example: Average Precision

```python
from deepfense.utils.registry import register_metric
import numpy as np
from sklearn.metrics import average_precision_score


@register_metric("AveragePrecision")
def compute_average_precision(labels, scores, params):
    """
    Compute Average Precision (area under precision-recall curve).
    """
    # Get positive class scores
    if scores.ndim == 2:
        if scores.shape[1] == 2:
            scores_1d = scores[:, 1]
        else:
            # For multi-class, use one-vs-rest
            return {"AveragePrecision": 0.0}  # Simplified
    else:
        scores_1d = scores
    
    try:
        ap = average_precision_score(labels, scores_1d)
        return {"AveragePrecision": ap}
    except ValueError:
        return {"AveragePrecision": 0.0}
```

## Example: Confusion Matrix Elements

```python
from deepfense.utils.registry import register_metric
import numpy as np
from sklearn.metrics import confusion_matrix


@register_metric("ConfusionMatrix")
def compute_confusion_matrix(labels, scores, params):
    """
    Compute confusion matrix and return individual elements.
    """
    threshold = params.get("threshold", 0.0)
    
    # Convert scores to predictions
    if scores.ndim == 2:
        predictions = np.argmax(scores, axis=1)
    else:
        predictions = (scores > threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # For binary classification, extract TP, TN, FP, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            "TruePositive": float(tp),
            "TrueNegative": float(tn),
            "FalsePositive": float(fp),
            "FalseNegative": float(fn)
        }
    else:
        # For multi-class, return the matrix (simplified)
        return {"ConfusionMatrix": cm.tolist()}
```

## Key Points

1. **Use @register_metric decorator**: Register with a unique string name
2. **Function signature**: Must accept `(labels, scores, params)` where:
   - `labels`: Ground truth labels as numpy array [N]
   - `scores`: Predicted scores as numpy array [N] (binary) or [N, C] (multi-class)
   - `params`: Dictionary with metric-specific parameters
3. **Return format**: Must return a dictionary `{"MetricName": value}`
4. **Handle score shapes**: Support both 1D (binary) and 2D (multi-class) scores
5. **Error handling**: Handle edge cases gracefully (e.g., all labels same class)
6. **No import needed**: Metrics are registered when the module is imported

## Function Signature

Your metric function should follow this pattern:

```python
@register_metric("MetricName")
def compute_metric(labels, scores, params):
    """
    Args:
        labels: Ground truth labels [N] - numpy array
        scores: Predicted scores [N] or [N, C] - numpy array
        params: Dictionary with metric parameters
    
    Returns:
        Dictionary with metric name and value: {"MetricName": float}
    """
    # Extract parameters
    param1 = params.get("param1", default_value)
    
    # Handle different score shapes
    if scores.ndim == 2:
        # Multi-class: use argmax or softmax
        predictions = np.argmax(scores, axis=1)
    else:
        # Binary: threshold at 0
        predictions = (scores > 0).astype(int)
    
    # Compute metric
    metric_value = compute_metric_value(labels, predictions)
    
    return {"MetricName": metric_value}
```

## Handling Score Shapes

Scores can come in different formats:

**Binary classification (1D):**
```python
scores = np.array([0.8, -0.3, 0.5, -0.1])  # [N]
predictions = (scores > 0).astype(int)  # [0, 1, 0, 1]
```

**Multi-class (2D):**
```python
scores = np.array([
    [0.2, 0.8],  # Sample 1: class 1
    [0.9, 0.1],  # Sample 2: class 0
])  # [N, C]
predictions = np.argmax(scores, axis=1)  # [1, 0]
```

## Testing Your Metric

Test your metric before using it in evaluation:

```python
import numpy as np
from deepfense.training.evaluations import metrics  # Import to register
from deepfense.utils.registry import METRIC_REGISTRY, build_metric

# Test data
labels = np.array([1, 0, 1, 0, 1])
scores_binary = np.array([0.8, -0.2, 0.9, -0.1, 0.7])  # Binary
scores_multiclass = np.array([
    [0.2, 0.8],
    [0.9, 0.1],
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7]
])  # Multi-class

# Test binary scores
params = {"threshold": 0.0}
metric_fn = METRIC_REGISTRY.get("Precision")
result_binary = metric_fn(labels, scores_binary, params)
print(f"Binary Precision: {result_binary}")

# Test multi-class scores
result_multiclass = metric_fn(labels, scores_multiclass, params)
print(f"Multi-class Precision: {result_multiclass}")

# Test with custom parameters
params_custom = {"threshold": 0.5, "average": "micro"}
result_custom = metric_fn(labels, scores_binary, params_custom)
print(f"Custom Precision: {result_custom}")
```

## Common Metrics in DeepFense

DeepFense already includes several metrics:

- **EER**: Equal Error Rate (see `compute_eer.py`)
- **minDCF**: minimum Detection Cost Function (see `compute_mindcf.py`)
- **ACC**: Accuracy
- **F1_SCORE**: F1 Score

You can add more metrics following the same pattern.

## Next Steps

- See [Training Guide](training_with_cli.md) for how metrics are used in training
- See [Configuration Reference](../05_configuration.md) for full config options
- See existing metrics in `deepfense/training/evaluations/metrics.py` for reference
- See `deepfense/training/evaluations/compute_eer.py` for a complex metric example

