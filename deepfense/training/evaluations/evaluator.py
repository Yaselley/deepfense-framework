import numpy as np
from deepfense.utils.registry import METRIC_REGISTRY


class Evaluator:
    """
    Evaluates a set of metrics defined in a configuration dictionary.

    Example config:
    {
        "EER": {},
        "ACC": {},
        "CLLR": {"bonafide_label": 1},
        "F1_SCORE": {"f1_average": "macro"}
    }
    """

    def __init__(self, config):
        self.config = config or {}
        self.metrics = self._load_metrics()

    def _clean_value(self, v):
        """Helper to convert numpy types to python types."""
        if isinstance(v, np.generic):
            # .item() converts np.float64 -> float, np.int32 -> int, etc.
            return v.item()
        return v

    def _load_metrics(self):
        """Load all metric functions from the global METRIC_REGISTRY."""
        metrics = {}
        for name in self.config:
            if name != "loss":  # skip loss
                try:
                    metric_fn = METRIC_REGISTRY.get(name)
                    metrics[name] = metric_fn
                except KeyError:
                    # Skip unknown metrics or raise error? Original raised error.
                    # raise ValueError(f"Unknown metric: '{name}' (not in METRIC_REGISTRY)")
                    print(f"[Warning] Metric '{name}' not found in registry. Skipping.")
        return metrics

    def evaluate(self, labels, scores):
        """
        Evaluate all registered metrics.
        Each metric is called with its own parameters + shared kwargs.

        Example:
            evaluator.evaluate(bonafide_scores=bona, spoof_scores=spoof)
        """
        results = {}
        for name, metric_fn in self.metrics.items():
            params = self.config.get(name, {})
            params["loss"] = self.config.get("loss")

            try:
                metric_result = metric_fn(labels, scores, params)
                if isinstance(metric_result, dict):
                    for k, v in metric_result.items():
                        v = self._clean_value(v)

                        if "threshold" not in k.lower():
                            results[k] = v
                else:
                    # Single float result
                    results[name] = metric_result

            except Exception as e:
                print(f"[Warning] Metric '{name}' failed: {e}")
                results[name] = None
        return results
