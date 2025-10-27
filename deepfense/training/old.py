# training/standard_trainer.py

import os
import time
import json
import math
import logging
from typing import Dict, Optional, Callable

import numpy as np
from sklearn import metrics as sk_metrics
from tqdm import tqdm

import torch
from torch import nn

from training.base_trainer import BaseTrainer
from training.registry import register_trainer
from training.sam import SAM


@register_trainer("standard")
class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer (your original Trainer), built on top of BaseTrainer.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: Callable,
        optimizer_config: Optional[dict],
        config,
    ):
        super().__init__(model, config)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        # optimizers / schedulers
        self.optimizer = self._build_optimizer(optimizer_config)
        if config.use_sam:
            self.optimizer = SAM(self.optimizer, self.model, rho=config.sam_rho)

        self.scheduler = self._build_scheduler(config.scheduler) if config.scheduler else None

        # output dirs
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
        opt_name = opt_cfg.get("name", "adam").lower()
        params = self.model.parameters()
        lr = opt_cfg.get("lr", 1e-4)
        wd = opt_cfg.get("weight_decay", 0.0)

        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=opt_cfg.get("momentum", 0.9), weight_decay=wd)

        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # ------------------------------
    # Scheduler Builder
    # ------------------------------
    def _build_scheduler(self, sched_cfg):
        name = sched_cfg.get("name", "").lower()
        opt = self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer

        if name == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.1)
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=sched_cfg.get("t_max", self.config.epochs)
            )
        if name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=sched_cfg.get("max_lr", 1e-3),
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.epochs
            )

        return None

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
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(images)
        if isinstance(logits, dict) and "cls" in logits:
            logits = logits["cls"]

        loss = self.criterion(logits, labels)

        if isinstance(self.optimizer, SAM):
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            logits2 = self.model(images)
            if isinstance(logits2, dict) and "cls" in logits2:
                logits2 = logits2["cls"]
            loss2 = self.criterion(logits2, labels)
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    # ------------------------------
    # Evaluation
    # ------------------------------
    def evaluate(self, epoch, step):
        self.model.eval()
        all_labels, all_probs, all_names = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                names = batch["dataset_name"]

                logits = self.model(imgs)
                if isinstance(logits, dict) and "cls" in logits:
                    logits = logits["cls"]

                if logits.dim() == 2 and logits.size(1) == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probs = logits.squeeze()

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                all_names.extend(names)

        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        names = np.array(all_names)

        results = {}
        results["global"] = self._compute_metrics(labels, probs)

        for ds in np.unique(names):
            mask = names == ds
            results[ds] = self._compute_metrics(labels[mask], probs[mask])

        # Save metrics JSON
        json_path = os.path.join(self.results_dir, f"metrics_epoch{epoch}_step{step}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        if self.wandb:
            for ds, metrics in results.items():
                self.wandb.log({f"{ds}/{k}": v for k, v in metrics.items()}, step=step)

        self.model.train()
        return results

    # ------------------------------
    # Metrics + Checkpointing
    # ------------------------------
    def _compute_metrics(self, labels, probs):
        preds = (probs >= 0.5).astype(int)
        return {
            "acc": sk_metrics.accuracy_score(labels, preds),
            "precision": sk_metrics.precision_score(labels, preds, zero_division=0),
            "recall": sk_metrics.recall_score(labels, preds, zero_division=0),
            "f1": sk_metrics.f1_score(labels, preds, zero_division=0),
            "auc": sk_metrics.roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
            "ap": sk_metrics.average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        }

    def _maybe_checkpoint(self, metrics: Dict, epoch: int, step: int):
        metric = metrics["global"][self.config.monitor_metric]
        better = (metric > self.best_metric) if self.config.monitor_mode == "max" else (metric < self.best_metric)
        if better:
            self.best_metric = metric
            self.save_checkpoint(epoch, step, is_best=True)

    def _current_lr(self):
        opt = self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer
        return opt.param_groups[0]["lr"]
