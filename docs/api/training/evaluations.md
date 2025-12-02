# Evaluator

`deepfense.training.evaluations.evaluator.Evaluator`

Computes standardized metrics for Anti-Spoofing.

## Supported Metrics

### EER (Equal Error Rate)
The threshold where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).

### minDCF (Minimum Detection Cost Function)
A weighted cost function standard in ASVSpoof challenges.

**Config**:
```yaml
metrics:
  minDCF:
    Pspoof: 0.05
    Cmiss: 1
    Cfa: 1
```

### Accuracy / F1
Standard classification metrics.

