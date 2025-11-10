# training/standard_trainer.py

import os
import json
import logging
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from deepfense.training.base_trainer import BaseTrainer
from deepfense.training.registry import register_trainer
from deepfense.training.optimizers.registry import OPTIMIZER_REGISTRY
from deepfense.training.schedulers.registry import SCHEDULER_REGISTRY
from deepfense.training.evaluations.evaluator import Evaluator

@register_trainer("StandardTrainer")
class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer (your original Trainer), built on top of BaseTrainer.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer_config: Optional[dict],
        scheduler_config: Optional[dict],
        metrics_config: Optional[dict],
        config,
    ):
        super().__init__(model, config)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizers / schedulers
        self.optimizer = self._build_optimizer(optimizer_config)
        self.scheduler = self._build_scheduler(self.optimizer, scheduler_config) if scheduler_config else None

        # evaluator
        metrics_config
        self.evaluator = Evaluator(metrics_config) if metrics_config else None

        # output dirs
        self.output_dir = config.output_dir
        self.results_dir = os.path.join(config.output_dir, "results")
        self.ckpts_dir = os.path.join(config.output_dir, "ckpts")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)

        # wandb
        if config.wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=config.wandb_project, config=config.__dict__)
        else:
            self.wandb = None

        # logger
        self.logger = logging.getLogger("trainer")

    # ------------------------------
    # Optimizer Builder
    # ------------------------------
    def _build_optimizer(self, opt_cfg):
        opt_name = opt_cfg.get("type", "adam").lower()
        optimizer_class = OPTIMIZER_REGISTRY[opt_name]
        params = self.model.parameters()
        return optimizer_class(params, opt_cfg.get("params", {}))

    # ------------------------------
    # Scheduler Builder
    # ------------------------------
    def _build_scheduler(self, optim, sched_cfg):
        sched_name = sched_cfg.get("type", "").lower()
        opt = self.optimizer

        if sched_name is None or sched_name == "":
            return None
        
        scheduler_class = SCHEDULER_REGISTRY[sched_name]
        return scheduler_class(optim, sched_cfg.get("params", {}))

    # ------------------------------
    # Training Loop
    # ------------------------------
    def train(self):
        self.model.train()

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters: {num_params:,}")

        for epoch in range(self.start_epoch, self.config.epochs):
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

            for batch_idx, batch in enumerate(loop):
                self.global_step += 1
                loss = self._train_step(batch)

                # Log
                if batch_idx % self.config.batch_log_interval == 0:
                    lr = self._current_lr()
                    self.logger.info(f"[Epoch {epoch}] [Step {batch_idx}] Loss={loss:.4f} LR={lr:.6f}")
                    if self.wandb:
                        self.wandb.log({"train/loss": loss, "lr": lr, "step": self.global_step})

                # Step-based eval
                if self.config.eval_every_steps and self.global_step % self.config.eval_every_steps == 0:
                    metrics = self.evaluate(epoch, self.global_step)
                    self._maybe_checkpoint(metrics, epoch, self.global_step)

                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    self.logger.info("Reached max steps; exiting.")
                    return

            # Per-epoch eval
            if self.config.eval_every_epochs and (epoch + 1) % self.config.eval_every_epochs == 0:
                metrics = self.evaluate(epoch, self.global_step)
                self._maybe_checkpoint(metrics, epoch, self.global_step)

            # Scheduler (except OneCycle)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

    # ------------------------------
    def _train_step(self, batch):
        """
        Training step for batches with:
            batch["x"], batch["label"], batch["mask"], batch["dataset_name"]
        """
        x = batch["x"].to(self.device)          # waveform or features
        labels = batch["label"].to(self.device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass through the detector
        outputs = self.model(x, mask=mask) if mask is not None else self.model(x)

        # Compute total loss using ModularDetector's compute_loss
        loss = self.model.compute_loss(outputs, labels)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss.item()


    # ------------------------------
    # Evaluation
    # ------------------------------
    def evaluate(self, epoch, step):
        self.model.eval()
        all_labels, all_probs, all_names, all_losses = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                x = batch["x"].to(self.device)
                labels = batch["label"].to(self.device)
                mask = batch.get("mask", None)
                names = batch["dataset_name"]

                logits = self.model(x, mask=mask) if mask is not None else self.model(x)
                probs = logits["probs"]

                batch_loss = self.model.compute_loss(logits, labels)
                all_losses.append(batch_loss.detach().cpu().item())

                if torch.is_tensor(probs):
                    probs = probs.detach().cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().numpy()

                all_labels.append(labels)
                all_probs.append(probs)
                all_names.extend(names)


        labels = np.concatenate(all_labels, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        names = np.array(all_names)

        results = {}
        results["loss"] = float(np.mean(all_losses))
        results["average"] = self._compute_metrics(labels, probs)

        for ds in np.unique(names):
            mask_ds = names == ds
            results[str(ds)] = self._compute_metrics(labels[mask_ds], probs[mask_ds])

        # Save metrics JSON
        json_path = os.path.join(self.results_dir, f"metrics_epoch{epoch}_step{step}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Log to wandb
        if self.wandb:
            for ds, metrics in results.items():
                self.wandb.log({f"{ds}/{k}": v for k, v in metrics.items()}, step=step)

        self.model.train()
        return results


    # ------------------------------
    # Metrics + Checkpointing
    # ------------------------------
    def _compute_metrics(self, labels, probs):
        if self.evaluator:
            results = self.evaluator.evaluate(labels, probs)
        else:
            results = {}
        return results

    def _maybe_checkpoint(self, metrics: Dict, epoch: int, step: int):
        print(metrics)
        metric = metrics[self.config.monitor_metric]
        better = (metric > self.best_metric) if self.config.monitor_mode == "max" else (metric < self.best_metric)
        if better:
            self.best_metric = metric
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
        prefix = f"ckpt_epoch{epoch:03d}_step{step:08d}"
        fname = os.path.join(self.ckpts_dir, f"{prefix}.pth")
        torch.save(state, fname)
        self.logger.info(f"Saved checkpoint: {fname}")

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
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
