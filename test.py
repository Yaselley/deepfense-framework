import os
import json
import argparse
import yaml
import logging
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# --- Import components from your project ---
from deepfense.data.data_utils import build_dataloader
from deepfense.models.registry import DETECTOR
from deepfense.training.evaluations.evaluator import Evaluator

def load_config(config_path):
    """Loads a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging_test(output_dir):
    """Setup logging for testing, saving to the checkpoint's folder."""
    log_file = os.path.join(output_dir, "test.log")
    
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
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

# --- MODIFIED: Function updated to save a single predictions.txt file ---
def run_evaluation(model, test_loader, evaluator, device, logger, output_dir):
    """
    Runs the evaluation loop, adapted from StandardTrainer.evaluate.
    Saves predictions to output_dir/results/predictions
    """
    model.eval()
    all_labels, all_scores, all_names, all_losses = [], [], [], []
    all_keys = [] # --- NEW: To store audio IDs
    
    logger.info("Starting evaluation on the test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["x"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask", None)
            names = batch["dataset_name"]
            
            # Common key names are 'key', 'keys', 'id', 'ids', 'audio_id'
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
        results["average"] = average_metrics # Fallback

    # Compute metrics for each dataset present in the test set
    for ds in np.unique(names):
        mask_ds = names == ds
        ds_labels = labels[mask_ds]
        ds_scores = scores[mask_ds]
        ds_keys = keys[mask_ds] if keys.size > 0 else []

        # Compute metrics
        results[str(ds)] = _compute_metrics(evaluator, ds_labels, ds_scores)

        # --- NEW: Save predictions to a single .txt file per dataset ---
        
        # 1. Handle placeholder IDs if keys were not found in batch
        if len(ds_keys) != len(ds_labels):
            if len(ds_keys) == 0:
                logger.warning(f"No 'keys' found in batch for dataset '{ds}'. Generating placeholder IDs.")
            else:
                logger.warning(f"Mismatched keys and labels for dataset '{ds}'. Generating placeholder IDs.")
            ds_keys = [f"{ds}_sample_{i:06d}" for i in range(len(ds_labels))]

        # 2. Process scores into class 0 and class 1
        scores_c0, scores_c1 = None, None
        if ds_scores.ndim == 1:
            # Assume score is for class 1 (positive)
            scores_c1 = ds_scores
            scores_c0 = 1.0 - ds_scores
        elif ds_scores.ndim == 2 and ds_scores.shape[1] == 2:
            # Assume [class_0, class_1]
            scores_c0 = ds_scores[:, 0]
            scores_c1 = ds_scores[:, 1]
        elif ds_scores.ndim == 2 and ds_scores.shape[1] == 1:
            # Assume [[class_1], [class_1], ...]
            scores_c1 = ds_scores.flatten()
            scores_c0 = 1.0 - scores_c1
        else:
            logger.error(f"Unsupported score shape {ds_scores.shape} for dataset '{ds}'. Cannot save predictions.")
            continue # Skip to next dataset

        # 3. Write the file
        prediction_file_path = os.path.join(predictions_dir, f"{str(ds)}_predictions.txt")
        try:
            with open(prediction_file_path, "w") as f:
                f.write("ID_audio,label,score_class0,score_class1\n") # Header
                for i in range(len(ds_labels)):
                    f.write(f"{ds_keys[i]},{int(ds_labels[i])},{scores_c0[i]:.8f},{scores_c1[i]:.8f}\n")
        except Exception as e:
            logger.warning(f"Failed to save prediction file for dataset '{ds}': {e}")
        # --- END NEW SECTION ---


    # --- Log results to console/file ---
    logger.info("--- Test Results ---")
    
    top_level_metrics = {}
    per_dataset_metrics = {}
    
    for ds_name, metric_values in results.items():
        if isinstance(metric_values, dict):
            per_dataset_metrics[ds_name] = metric_values
        else:
            top_level_metrics[ds_name] = metric_values

    avg_metrics_str = ", ".join([
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
        for k, v in top_level_metrics.items()
    ])
    logger.info(f"📈 Overall Metrics: {avg_metrics_str}")

    for ds_name, metrics_dict in per_dataset_metrics.items():
        ds_metrics_str = ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
            for k, v in metrics_dict.items()
        ])
        logger.info(f"📊 Dataset '{ds_name}': {ds_metrics_str}")
    logger.info("------------------------")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run testing from a config and checkpoint.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the YAML config file (e.g., config.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the model checkpoint file (e.g., best_model.pth)")
    args = parser.parse_args()

    # --- 1. Setup ---
    # Determine output folder (same as checkpoint folder)
    output_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, "results.json")
    
    # Setup logging
    logger = setup_logging_test(output_dir)
    logger.info(f"Loading config from: {args.config}")
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    
    # Load config
    cfg = load_config(args.config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Build Model ---
    detector_cfg = cfg["detector"]
    detector_cfg["loss"] = cfg["loss"] # Add loss config as train.py does
    model = DETECTOR[detector_cfg["type"]](detector_cfg)
    model.to(device)

    # --- 3. Load Checkpoint ---
    try:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        logger.info(f"Successfully loaded model state from epoch {state.get('epoch', 'N/A')}, step {state.get('step', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    # --- 4. Build Test Dataloader ---
    try:
        test_cfg = cfg["data"]["test"]
        test_cfg["label_map"] = cfg["data"]["label_map"]
        test_cfg["sampling_rate"] = cfg["data"]["sampling_rate"]
    except KeyError:
        logger.error("Could not find 'data.test' section in config file.")
        return
        
    test_loader = build_dataloader(test_cfg)
    logger.info(f"Test dataloader built successfully.")

    # --- 5. Build Evaluator ---
    metrics_config = cfg.get("metrics", None)
    if metrics_config:
        metrics_config["loss"] = model.main_loss.lower()
        evaluator = Evaluator(metrics_config)
        logger.info("Evaluator built successfully.")
    else:
        evaluator = None
        logger.warning("No 'metrics' config found. Only loss will be reported.")

    # --- 6. Run Evaluation ---
    results = run_evaluation(model, test_loader, evaluator, device, logger, output_dir)

    # --- 7. Save Results ---
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results successfully saved to: {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results.json: {e}")

if __name__ == "__main__":
    main()