# Visualization Utilities

`deepfense.utils.visualization`

Helper functions for plotting training progress.

## Functions

### `plot_metric_trend`

Plots the history of a metric (e.g., Loss, EER) over epochs/steps.

*   **Args**:
    *   `history_dict`: Dictionary mapping split names (Train/Val) to list of `(step, value)` tuples.
    *   `metric_name`: Name of the metric (title).
    *   `save_path`: Output filename.
    *   `xlabel`: "Epoch" or "Step".

