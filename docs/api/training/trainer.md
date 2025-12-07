# StandardTrainer

`deepfense.training.standard_trainer.StandardTrainer`

Manages the supervised training loop.

## Key Responsibilities

1.  **Initialization**: Builds optimizer, scheduler, and evaluator.
2.  **Training Loop**: Iterates through epochs and batches.
3.  **Gradient Accumulation**: Supports `accum_steps` in config.
4.  **Logging**: Logs to console and WandB (if enabled).
5.  **Evaluation**: Runs validation every `eval_every_epochs` or `eval_every_steps`.
6.  **Checkpointing**: Saves `best_model.pth` based on `monitor_metric`.
7.  **Early Stopping**: Stops if metric doesn't improve for `patience` epochs.

## Methods

*   `train()`: Main entry point.
*   `evaluate(epoch, step)`: Runs validation loop.
*   `save_checkpoint(...)`: Saves state dicts.
*   `load_checkpoint(...)`: Resumes training.

