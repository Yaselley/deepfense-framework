import os
import json
import argparse
import logging
from datetime import datetime
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm

from deepfense.data.data_utils import build_dataloader
from deepfense.utils.registry import build_detector
from deepfense.models import * 
from deepfense.training.evaluations.evaluator import Evaluator
from deepfense.training.evaluations.utils import _metric_get_1d_scores
from deepfense.utils.predictions_io import (
    append_frame_predictions,
    build_eval_metadata,
    save_clip_predictions_jsonl,
    save_frame_predictions_jsonl,
)


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


def _compute_metrics(evaluator, labels, scores, keys=None, label_hop_ms=None, label_merge_rule=None):
    """Helper to run the evaluator."""
    if evaluator:
        ctx = {}
        if keys is not None:
            ctx["keys"] = keys
            ctx["temporal"] = True
        if label_hop_ms is not None:
            ctx["label_hop_ms"] = float(label_hop_ms)
        if label_merge_rule is not None:
            ctx["label_merge_rule"] = str(label_merge_rule)
        return evaluator.evaluate(labels, scores, **ctx)
    return {}


def run_evaluation(
    model,
    test_loader,
    evaluator,
    device,
    logger,
    output_dir,
    label_hop_ms=None,
    label_hop_samples=None,
    sampling_rate=None,
    label_merge_rule=None,
):
    """
    Runs the evaluation loop.
    Saves predictions to output_dir/results/predictions
    """
    model.eval()
    all_labels, all_scores, all_names, all_losses = [], [], [], []
    all_keys = []
    frame_pred_store = {}
    temporal_eval = False

    if label_hop_ms is None:
        label_hop_ms = 20.0
    hop_ms = float(label_hop_ms)

    logger.info("Starting evaluation on the test set...")

    if len(model.losses):
        bona = model.losses[model.main_loss_idx].bonafide_label
    else:
        bona = 1

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["x"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask", None)
            names = batch["dataset_name"]
            keys = batch["ID"]
            frame_bt = batch.get("frame_labels")

            outputs = model(x, mask=mask) if mask is not None else model(x)
            scores = outputs["scores"]

            if frame_bt is not None:
                temporal_eval = True
                fl = frame_bt.to(device)
                batch_loss = model.compute_loss(outputs, fl)
                logits_t = outputs["logits"]
                if logits_t is None:
                    raise RuntimeError("Temporal test requires outputs['logits'].")
                logits_np = logits_t.detach().cpu().numpy()
                fl_np = fl.detach().cpu().numpy()
                fm = batch.get("frame_mask")
                fm_np = fm.numpy() if fm is not None else np.ones_like(fl_np, dtype=np.float32)
                B, T_log, C = logits_np.shape
                flat_logits = logits_np.reshape(-1, C)
                flat_scores = _metric_get_1d_scores(
                    flat_logits, {"loss": "crossentropy", "bonafide_label": bona}
                )
                scores_bt = flat_scores.reshape(B, T_log)
                T_min = min(scores_bt.shape[1], fl_np.shape[1], fm_np.shape[1])
                scores_bt = scores_bt[:, :T_min]
                fl_np = fl_np[:, :T_min]
                fm_np = fm_np[:, :T_min]
                valid = (fl_np != -100) & (fm_np > 0)
                append_frame_predictions(
                    frame_pred_store,
                    utt_ids=keys,
                    dataset_names=names,
                    frame_labels=fl_np,
                    logits=logits_np[:, :T_min, :],
                    valid_mask=valid,
                    label_hop_ms=hop_ms,
                    label_hop_samples=label_hop_samples,
                    bonafide_label=bona,
                )
                all_labels.append(fl_np[valid])
                all_scores.append(scores_bt[valid])
                for i in range(B):
                    n_valid = int(valid[i].sum())
                    all_names.extend([names[i]] * n_valid)
                    all_keys.extend([keys[i]] * n_valid)
            else:
                batch_loss = model.compute_loss(outputs, labels)
                if torch.is_tensor(scores):
                    scores = scores.detach().cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().numpy()
                all_labels.append(labels)
                all_scores.append(scores)
                all_names.extend(names)
                all_keys.extend(keys)

            all_losses.append(batch_loss.detach().cpu().item())
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
    results["evaluation"] = build_eval_metadata(
        temporal=temporal_eval,
        label_hop_ms=hop_ms if temporal_eval else None,
        label_hop_samples=label_hop_samples if temporal_eval else None,
        sampling_rate=sampling_rate,
    )
    results["loss"] = float(np.mean(all_losses))

    # Compute average metrics over all datasets
    average_metrics = _compute_metrics(
        evaluator, labels, scores,
        keys=keys if temporal_eval else None,
        label_hop_ms=label_hop_ms,
        label_merge_rule=label_merge_rule,
    )
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
        results[str(ds)] = _compute_metrics(
            evaluator, ds_labels, ds_scores,
            keys=keys[mask_ds] if temporal_eval else None,
            label_hop_ms=label_hop_ms,
            label_merge_rule=label_merge_rule,
        )

        prediction_file_path = os.path.join(
            predictions_dir, f"{str(ds)}_predictions.jsonl"
        )
        try:
            if temporal_eval:
                ds_utts = frame_pred_store.get(ds, {})
                save_frame_predictions_jsonl(
                    prediction_file_path,
                    ds_utts,
                    metadata=results["evaluation"],
                )
            else:
                if len(ds_keys) != len(ds_labels):
                    ds_keys = [f"{ds}_sample_{i:06d}" for i in range(len(ds_labels))]
                save_clip_predictions_jsonl(
                    prediction_file_path,
                    ds_keys,
                    ds_labels,
                    ds_scores,
                    metadata={"temporal": False},
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
    logger.info(f"📈 Overall Metrics: {avg_metrics_str}")

    for ds_name, metrics_dict in per_dataset_metrics.items():
        ds_metrics_str = ", ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()]
        )
        logger.info(f"📊 Dataset '{ds_name}': {ds_metrics_str}")
    logger.info("------------------------")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run testing from a config and checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, "results.json")

    logger = setup_logging_test(output_dir)
    logger.info(f"Loading config from: {args.config}")
    logger.info(f"Loading checkpoint from: {args.checkpoint}")

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_detector(cfg.model.type, model_cfg)
    model.to(device)

    try:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)
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
        for hop_key in (
            "label_hop", "label_hop_ms",
            "source_label_hop", "source_label_hop_ms",
            "label_merge_rule",
        ):
            if hop_key in cfg.data and cfg.data.get(hop_key) is not None:
                test_cfg[hop_key] = cfg.data[hop_key]
    except Exception:
        logger.error("Could not configure test dataset.")
        return

    test_loader = build_dataloader(test_cfg)
    logger.info(f"Test dataloader built successfully.")

    metrics_config = OmegaConf.to_container(cfg.training.metrics, resolve=True) if "metrics" in cfg.training else None
    evaluator = Evaluator(metrics_config) if metrics_config else None

    label_hop_ms = None
    label_hop_samples = None
    sampling_rate = float(cfg.data.get("sampling_rate", 16000))
    if "label_hop_ms" in cfg.data and cfg.data.get("label_hop_ms") is not None:
        label_hop_ms = float(cfg.data.label_hop_ms)
    elif "label_hop" in cfg.data and cfg.data.get("label_hop") is not None:
        label_hop_samples = int(cfg.data.label_hop)
        label_hop_ms = label_hop_samples * 1000.0 / sampling_rate
    if label_hop_ms is not None and label_hop_samples is None:
        label_hop_samples = int(round(label_hop_ms * sampling_rate / 1000.0))
    label_merge_rule = cfg.data.get("label_merge_rule") if "label_merge_rule" in cfg.data else None

    results = run_evaluation(
        model, test_loader, evaluator, device, logger, output_dir,
        label_hop_ms=label_hop_ms,
        label_hop_samples=label_hop_samples,
        sampling_rate=sampling_rate,
        label_merge_rule=label_merge_rule,
    )

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results successfully saved to: {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results.json: {e}")


if __name__ == "__main__":
    main()
