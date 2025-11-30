# training/standard_trainer.py

import os
import json
import logging
from typing import Dict, Optional
import collections

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from deepfense.training.base_trainer import BaseTrainer
from deepfense.utils.registry import register_trainer
from deepfense.training.evaluations.evaluator import Evaluator
from deepfense.utils.visualization import plot_metric_trend


@register_trainer("StandardTrainer")
class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: DictConfig,
    ):
        """
        Args:
            model: The ModularDetector model.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            config: The 'training' section of the config (DictConfig).
        """
        super().__init__(model, config)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Output dirs (inherited/setup in BaseTrainer, but specialized here)
        self.results_dir = os.path.join(self.config.output_dir, "results")
        self.ckpts_dir = os.path.join(self.config.output_dir, "ckpts")
        self.plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Optimizers / Schedulers
        self.optimizer = self._build_optimizer(config.optimizer)
        self.scheduler = (
            self._build_scheduler(config.scheduler)
            if config.get("scheduler")
            else None
        )

        # Evaluator
        metrics_config = config.get("metrics", None)
        if metrics_config and hasattr(self.model, "main_loss_type"):
             metrics_config["loss"] = self.model.main_loss_type
             
        self.evaluator = Evaluator(metrics_config) if metrics_config else None

        # History tracking
        # Structure: self.metric_history[metric_name][split_name] = [(epoch, val), ...]
        self.metric_history = collections.defaultdict(lambda: collections.defaultdict(list))

        # WandB
        if self.config.get("wandb", False):
            import wandb

            self.wandb = wandb
            wandb_config = OmegaConf.to_container(config, resolve=True)
            wandb.init(
                project=self.config.get("wandb_project", "DeepFense"), 
                config=wandb_config
            )
        else:
            self.wandb = None

        self.logger = logging.getLogger("trainer")
        
        # Visualization Unit
        self.use_steps_for_viz = (self.config.get("eval_every_steps") is not None)
        self.viz_unit = "Step" if self.use_steps_for_viz else "Epoch"

    def train(self):
        self.model.train()

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters: {num_params:,}")

        for epoch in range(self.start_epoch, self.config.epochs):
            current_epoch = epoch + 1

            loop = tqdm(
                self.train_loader, desc=f"Epoch {current_epoch}/{self.config.epochs}"
            )

            epoch_loss_sum = 0.0
            epoch_train_losses = []

            for batch_idx, batch in enumerate(loop):
                self.global_step += 1
                loss = self._train_step(batch)

                epoch_loss_sum += loss
                epoch_train_losses.append(loss)

                # Logging
                if (
                    self.config.get("batch_log_interval") is not None
                    and batch_idx != 0
                    and batch_idx % self.config.batch_log_interval == 0
                ):
                    lr = self._current_lr()
                    running_avg_loss = epoch_loss_sum / (batch_idx + 1)

                    self.logger.info(
                        f"[Epoch {current_epoch}] [Step {batch_idx}] Running Avg Loss={running_avg_loss:.4f} LR={lr:.6f}"
                    )
                    if self.wandb:
                        self.wandb.log(
                            {
                                "train/running_avg_loss": running_avg_loss,
                                "lr": lr,
                                "step": self.global_step,
                            }
                        )
                    
                    # If in Step-based mode, record Train Loss here for finer granularity
                    if self.use_steps_for_viz:
                        self.metric_history["loss"]["Train"].append((self.global_step, running_avg_loss))


                # Step-based eval
                if (
                    self.config.get("eval_every_steps")
                    and self.global_step % self.config.eval_every_steps == 0
                ):
                    metrics = self.evaluate(
                        current_epoch, self.global_step, eval_reason="step"
                    )
                    self._maybe_checkpoint(metrics, current_epoch, self.global_step)

                if self.config.get("max_steps") and self.global_step >= self.config.max_steps:
                    self.logger.info("Reached max steps; exiting.")
                    return

            # Epoch Summary
            avg_epoch_loss = np.mean(epoch_train_losses)
            self.logger.info(f"--- Epoch {current_epoch} Summary ---")
            self.logger.info(f"Average Train Loss: {avg_epoch_loss:.4f}")
            
            # Track Train Loss (Only if in Epoch mode to avoid double tracking or mixed scales)
            if not self.use_steps_for_viz:
                self.metric_history["loss"]["Train"].append((current_epoch, avg_epoch_loss))
            
            if self.wandb:
                self.wandb.log(
                    {"train/epoch_loss": avg_epoch_loss, "epoch": current_epoch},
                    step=self.global_step,
                )

            # Per-epoch eval
            eval_interval = self.config.get("eval_every_epochs", 1)
            if current_epoch % eval_interval == 0:
                metrics = self.evaluate(
                    current_epoch, self.global_step, eval_reason="epoch"
                )
                self._maybe_checkpoint(metrics, current_epoch, self.global_step)

            # Scheduler
            if self.scheduler and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

    def _train_step(self, batch):
        x = batch["x"].to(self.device)
        labels = batch["label"].to(self.device)
        mask = batch.get("mask", None)

        # Handle 'concat' augmentation (x: [B, N_aug, T])
        if x.ndim == 3 and labels.shape[0] == x.shape[0]:
             B, N, T = x.shape
             # DEBUG PRINT
             if self.global_step < 5:
                 self.logger.info(f"[ConcatAug] Detected 3D input: {x.shape}. Expanding to ({B*N}, {T}).")

             x = x.view(B * N, T)
             labels = labels.repeat_interleave(N)
             
             if mask is not None:
                 if mask.ndim == 3:
                     mask = mask.view(B * N, T)
                 elif mask.ndim == 2:
                     mask = mask.repeat_interleave(N, dim=0)

        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(x, mask) if mask is not None else self.model(x)
        loss = self.model.compute_loss(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, epoch, step, eval_reason: str = None):
        self.model.eval()
        all_labels, all_scores, all_names, all_losses = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                x = batch["x"].to(self.device)
                labels = batch["label"].to(self.device)
                mask = batch.get("mask", None)
                names = batch["dataset_name"]

                outputs = (
                    self.model(x, mask=mask) if mask is not None else self.model(x)
                )
                scores = outputs["scores"]
                
                batch_loss = self.model.compute_loss(outputs, labels)
                all_losses.append(batch_loss.detach().cpu().item())

                if torch.is_tensor(scores):
                    scores = scores.detach().cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().numpy()

                all_labels.append(labels)
                all_scores.append(scores)
                all_names.extend(names)

        labels = np.concatenate(all_labels, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        names = np.array(all_names)

        results = {}
        results["loss"] = float(np.mean(all_losses))

        average_metrics = self._compute_metrics(labels, scores)
        if isinstance(average_metrics, dict):
            results.update(average_metrics)
        else:
            results["average"] = average_metrics

        for ds in np.unique(names):
            mask_ds = names == ds
            results[str(ds)] = self._compute_metrics(labels[mask_ds], scores[mask_ds])

        # --- History Tracking & Trend Plotting ---
        # Use Step or Epoch as x-axis
        x_val = step if self.use_steps_for_viz else epoch
        self._update_history(x_val, results)
        self._plot_metric_trends(x_val)

        # Logging to Console & JSON
        self._log_results(epoch, step, eval_reason, results)

        if self.wandb:
            self._log_wandb(step, results)

        self.model.train()
        return results

    def _update_history(self, x_val, results):
        # 1. Val Loss
        if "loss" in results:
             self.metric_history["loss"]["Val"].append((x_val, results["loss"]))
             
        # 2. Other Scalar Metrics (Top-level only, representing Average)
        ignore_keys = ["loss", "average"] 
        
        for key, val in results.items():
            if key in ignore_keys: 
                continue
            
            # Only track scalars (Averages)
            # We skip dictionaries (per-dataset metrics) to avoid duplicate plots like 'Val_EER'
            if isinstance(val, (int, float, np.number)):
                self.metric_history[key]["Val"].append((x_val, float(val)))

    def _plot_metric_trends(self, x_val):
        # Plot each metric history (Train vs Val comparison supported)
        for metric_name, history_dict in self.metric_history.items():
            if not history_dict:
                continue
            
            # Save locally
            save_path = os.path.join(self.plots_dir, f"trend_{metric_name}.png")
            
            # Use updated visualization util that supports dict {Split: history}
            plot_metric_trend(history_dict, metric_name, save_path, xlabel=self.viz_unit)
            
            # Log to WandB if enabled
            if self.wandb:
                self.wandb.log({f"plots/{metric_name}": self.wandb.Image(save_path)}, step=self.global_step)

    def _log_results(self, epoch, step, eval_reason, results):
        if eval_reason == "step":
            title = f"--- 🏃 Mid-Epoch Validation (Epoch {epoch}, Step {step}) ---"
        elif eval_reason == "epoch":
            title = f"--- 🏁 End-of-Epoch Validation (Epoch {epoch}) ---"
        else:
            title = f"--- Validation Results (Epoch {epoch}, Step {step}) ---"

        self.logger.info(title)

        top_level_metrics = {}
        per_dataset_metrics = {}

        for ds_name, metric_values in results.items():
            if isinstance(metric_values, dict):
                per_dataset_metrics[ds_name] = metric_values
            else:
                top_level_metrics[ds_name] = metric_values

        avg_metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in top_level_metrics.items()
            ]
        )
        self.logger.info(f"📈 Average Metrics: {avg_metrics_str}")

        for ds_name, metrics_dict in per_dataset_metrics.items():
            ds_metrics_str = ", ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics_dict.items()
                ]
            )
            self.logger.info(f"📊 Dataset '{ds_name}': {ds_metrics_str}")
        self.logger.info("--------------------------------------------------")

        json_path = os.path.join(
            self.results_dir, f"metrics_epoch{epoch}_step{step}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    def _log_wandb(self, step, results):
        top_level_metrics = {}
        per_dataset_metrics = {}
        for ds_name, metric_values in results.items():
            if isinstance(metric_values, dict):
                per_dataset_metrics[ds_name] = metric_values
            else:
                top_level_metrics[ds_name] = metric_values

        self.wandb.log(top_level_metrics, step=step)
        for ds_name, metrics_dict in per_dataset_metrics.items():
            self.wandb.log(
                {f"{ds_name}/{k}": v for k, v in metrics_dict.items()}, step=step
            )

    def _compute_metrics(self, labels, scores):
        if self.evaluator:
            results = self.evaluator.evaluate(labels, scores)
        else:
            results = {}
        return results

    def _maybe_checkpoint(self, metrics: Dict, epoch: int, step: int):
        metric = metrics
        monitor_metric = self.config.get("monitor_metric", "loss")
        monitor_mode = self.config.get("monitor_mode", "min")

        try:
            current_metric = metric
            for key in monitor_metric.split("."):
                current_metric = current_metric[key]
        except (KeyError, TypeError):
            self.logger.error(
                f"Could not find monitor_metric '{monitor_metric}' in metrics dict."
            )
            return 

        better = (
            (current_metric > self.best_metric)
            if monitor_mode == "max"
            else (current_metric < self.best_metric)
        )
        if better:
            self.best_metric = current_metric
            self.save_checkpoint(epoch, step, is_best=True)

    def _current_lr(self):
        opt = self.optimizer
        return opt.param_groups[0]["lr"]

    def save_checkpoint(self, epoch, step, is_best=False):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": (self.optimizer.state_dict()),
            "epoch": epoch,
            "step": step,
            "best_metric": self.best_metric,
        }
        prefix = f"ckpt_epoch{epoch:03d}_step{step:06d}"
        fname = os.path.join(self.ckpts_dir, f"{prefix}.pth")
        torch.save(state, fname)
        self.logger.info(f"Saved checkpoint: {fname}")

        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saved BEST checkpoint: {best_path}")

        return fname

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if load_optimizer:
            opt_state = state.get("optimizer_state", None)
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
        self.start_epoch = state.get("epoch", 0)
        self.global_step = state.get("step", 0)
        self.best_metric = state.get("best_metric", self.best_metric)
        self.logger.info(f"Loaded checkpoint from {path}")

    def infer(self, images):
        self.model.eval()
        return self.model(images)
