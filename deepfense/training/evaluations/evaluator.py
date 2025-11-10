from deepfense.training.evaluations.registry import EVAL_REGISTRY

class Evaluator:
    """
    Evaluates a set of metrics defined in a configuration dictionary.

    Example config:
    {
        "metrics": {
            "actDCF": {"Pspoof": 0.05, "Cmiss": 1, "Cfa": 1},
            "minDCF": {"Pspoof": 0.05, "Cmiss": 1, "Cfa": 1},
        }
    }
    """

    def __init__(self, config):
        self.config = config or {}
        print(self.config)
        self.metrics = self._load_metrics()

    def _load_metrics(self):
        """Load all metric functions from the global EVAL_REGISTRY."""
        metrics = {}
        for name in self.config:
            metric_fn = EVAL_REGISTRY.get(name)
            if metric_fn is None:
                raise ValueError(f"Unknown metric: '{name}' (not in EVAL_REGISTRY)")
            metrics[name] = metric_fn
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
            try:
                metric_result = metric_fn(labels, scores, params)
                if isinstance(metric_result, dict):
                    for k, v in metric_result.items():
                        if "threshold" not in k.lower():
                            results[k] = v
                else:
                    # Single float result
                    results[metric_fn.__name__] = metric_result

            except Exception as e:
                print(f"[Warning] Metric '{metric_fn.__name__}' failed: {e}")
                results[metric_fn.__name__] = None
        return results
