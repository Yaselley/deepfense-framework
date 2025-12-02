# Training Workflow

This guide explains the training process in detail.

## The Training Loop

The `StandardTrainer` executes the following loop:

1.  **Setup**:
    *   Loads data loaders.
    *   Builds model, optimizer, scheduler.
    *   Initializes WandB (if configured).

2.  **Epoch Loop**:
    *   **Train Step**:
        *   Load batch.
        *   Apply augmentations (if not applied in dataset).
        *   Forward pass through `Detector`.
        *   Compute Loss (weighted sum of all losses).
        *   Backprop & Optimizer Step.
    *   **Logging**:
        *   Every `batch_log_interval`, logs running loss to console/WandB.
    *   **Evaluation**:
        *   Every `eval_every_epochs` (or steps).
        *   Computes EER, minDCF on Validation set.
        *   Saves checkpoint if metric improves.

## Checkpoints

Checkpoints are saved in `outputs/EXP_NAME/ckpts/`.

*   `ckpt_epochXX_stepYY.pth`: Periodic checkpoints.
*   `best_model.pth`: The best model according to `monitor_metric`.

## Resuming Training

To resume, you currently need to modify the code or config to load a specific checkpoint, or implement a `resume_from` flag in `train.py` (see `tutorials/extending.md`).

