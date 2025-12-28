"""Testing command for DeepFense."""
import os
import json
import logging
from omegaconf import OmegaConf
import click
import torch
from tqdm import tqdm
import numpy as np

from deepfense.data.data_utils import build_dataloader
from deepfense.utils.registry import build_detector
from deepfense.models import * 
from deepfense.training.evaluations.evaluator import Evaluator


def load_config(config_path):
    """Loads a YAML config file."""
    return OmegaConf.load(config_path)


def setup_logging_test(output_dir):
    """Setup logging for testing, saving to the checkpoint's folder."""
    log_file = os.path.join(output_dir, "test.log")

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Test logging configured. Log file: {log_file}")
    return logger


def _compute_metrics(evaluator, labels, scores):
    """Helper to run the evaluator."""
    if evaluator:
        return evaluator.evaluate(labels, scores)
    return {}


def run_evaluation(model, test_loader, evaluator, device, logger, output_dir):
    """
    Runs the evaluation loop.
    Saves predictions to output_dir/results/predictions
    """
    model.eval()
    all_labels, all_scores, all_names, all_losses = [], [], [], []
    all_keys = [] 

    logger.info("Starting evaluation on the test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["x"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask", None)
            names = batch["dataset_name"]
            keys = batch["ID"]

            outputs = model(x, mask=mask) if mask is not None else model(x)
            scores = outputs["scores"]

            # Compute loss for this batch
            batch_loss = model.compute_loss(outputs, labels)
            all_losses.append(batch_loss.detach().cpu().item())

            # Detach and move to CPU
            if torch.is_tensor(scores):
                scores = scores.detach().cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()

            all_labels.append(labels)
            all_scores.append(scores)
            all_names.extend(names)
            all_keys.extend(keys)
    # Concatenate all results
    labels = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    names = np.array(all_names)
    keys = np.array(all_keys)

    # --- Setup predictions directory ---
    predictions_dir = os.path.join(output_dir, "results", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    logger.info(f"Saving per-dataset predictions to: {predictions_dir}")

    results = {}
    results["loss"] = float(np.mean(all_losses))

    # Compute average metrics over all datasets
    average_metrics = _compute_metrics(evaluator, labels, scores)
    if isinstance(average_metrics, dict):
        results.update(average_metrics)
    else:
        results["average"] = average_metrics  # Fallback

    # Compute metrics for each dataset present in the test set
    for ds in np.unique(names):
        mask_ds = names == ds
        ds_labels = labels[mask_ds]
        ds_scores = scores[mask_ds]
        ds_keys = keys[mask_ds] if keys.size > 0 else []

        # Compute metrics
        results[str(ds)] = _compute_metrics(evaluator, ds_labels, ds_scores)

        # --- Save predictions to a single .txt file per dataset ---
        if len(ds_keys) != len(ds_labels):
            ds_keys = [f"{ds}_sample_{i:06d}" for i in range(len(ds_labels))]

        scores_c0, scores_c1 = None, None
        if ds_scores.ndim == 1:
            scores_c1 = ds_scores
            scores_c0 = 1.0 - ds_scores
        elif ds_scores.ndim == 2 and ds_scores.shape[1] == 2:
            scores_c0 = ds_scores[:, 0]
            scores_c1 = ds_scores[:, 1]
        elif ds_scores.ndim == 2 and ds_scores.shape[1] == 1:
            scores_c1 = ds_scores.flatten()
            scores_c0 = 1.0 - scores_c1
        else:
            continue 

        prediction_file_path = os.path.join(
            predictions_dir, f"{str(ds)}_predictions.txt"
        )
        try:
            with open(prediction_file_path, "w") as f:
                f.write("ID_audio,label,score_class0,score_class1\n")
                for i in range(len(ds_labels)):
                    f.write(
                        f"{ds_keys[i]},{int(ds_labels[i])},{scores_c0[i]:.8f},{scores_c1[i]:.8f}\n"
                    )
        except Exception as e:
            logger.warning(f"Failed to save prediction file for dataset '{ds}': {e}")

    # --- Log results ---
    logger.info("--- Test Results ---")
    top_level_metrics = {}
    per_dataset_metrics = {}

    for ds_name, metric_values in results.items():
        if isinstance(metric_values, dict):
            per_dataset_metrics[ds_name] = metric_values
        else:
            top_level_metrics[ds_name] = metric_values

    avg_metrics_str = ", ".join(
        [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in top_level_metrics.items()]
    )
    logger.info(f"Overall Metrics: {avg_metrics_str}")

    for ds_name, metrics_dict in per_dataset_metrics.items():
        ds_metrics_str = ", ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()]
        )
        logger.info(f"Dataset '{ds_name}': {ds_metrics_str}")
    logger.info("------------------------")

    return results


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--checkpoint", "-ckpt", required=True, type=click.Path(exists=True), help="Path to model checkpoint file")
def test(config, checkpoint):
    """
    Test a trained DeepFense model.
    
    Example:
    
        deepfense test --config config/train.yaml --checkpoint outputs/exp/best_model.pth
    """
    output_dir = os.path.dirname(checkpoint)
    results_path = os.path.join(output_dir, "results.json")

    logger = setup_logging_test(output_dir)
    logger.info(f"Loading config from: {config}")
    logger.info(f"Loading checkpoint from: {checkpoint}")

    cfg = load_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_detector(cfg.model.type, model_cfg)
    model.to(device)

    try:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        logger.info(f"Successfully loaded model state.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    try:
        test_cfg = OmegaConf.to_container(cfg.data.test, resolve=True)
        # Inject global data settings
        if "label_map" in cfg.data:
            test_cfg["label_map"] = OmegaConf.to_container(cfg.data.label_map, resolve=True)
        if "sampling_rate" in cfg.data:
            test_cfg["sampling_rate"] = cfg.data.sampling_rate
    except Exception:
        logger.error("Could not configure test dataset.")
        return

    test_loader = build_dataloader(test_cfg)
    logger.info(f"Test dataloader built successfully.")

    metrics_config = OmegaConf.to_container(cfg.training.metrics, resolve=True) if "metrics" in cfg.training else None
    evaluator = Evaluator(metrics_config) if metrics_config else None

    results = run_evaluation(model, test_loader, evaluator, device, logger, output_dir)

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results successfully saved to: {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results.json: {e}")

