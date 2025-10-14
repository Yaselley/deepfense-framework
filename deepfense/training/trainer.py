import os
import time
import yaml
import math
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np 
from sklearn import metrics as sk_metrics
import json 
from tqdm import tqdm

import torch
from torch import nn
from torch.cuda.amp import autocast

# Optional wandb
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

logger = logging.getLogger(__name__)


class SAM:
    """
    Sharpness-Aware Minimization (SAM) wrapper.
    Usage: base_opt = torch.optim.SGD(model.parameters(), lr=...); opt = SAM(base_opt)
    Call opt.first_step(zero_grad=True); then loss2.backward(); opt.second_step(zero_grad=True)
    """
    def __init__(self, base_optimizer, model, rho=0.05, adaptive=False):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.model = model
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        eps = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.adaptive:
                    scale = (p.abs() + 1e-12)
                    e = grad * (group.get("lr", 0.0) * self.rho) * scale
                else:
                    e = grad * (group.get("lr", 0.0) * self.rho)
                p.add_(e)
                eps[p] = e
        if zero_grad:
            self.zero_grad()
        self._eps = eps

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        # restore params and step base optimizer
        for p, e in self._eps.items():
            p.sub_(e)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self):
        # fallback for single-step optimizers -- call base optimizer
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

@dataclass
class TrainerConfig:
    output_dir: str
    device: str = "cuda"
    epochs: int = 10
    max_steps: Optional[int] = None
    eval_every_steps: Optional[int] = None
    eval_every_epochs: Optional[int] = 1
    save_every_epochs: Optional[int] = 1
    batch_log_interval: int = 50
    seed: Optional[int] = None
    use_amp: bool = False
    wandb: bool = False
    wandb_project: Optional[str] = None
    optimizer: str = "adam"
    scheduler: Optional[dict] = None
    use_sam: bool = False
    sam_rho: float = 0.05
    monitor_metric: str = "auc"  # metric name to select best checkpoint
    monitor_mode: str = "max"  # 'max' or 'min'
    batch_size: int=32
    num_workers: int=4

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: Callable,
        optimizer_config: Optional[dict],
        config: TrainerConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = self.config.device
        self.model.to(self.device)
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -math.inf if config.monitor_mode == "max" else math.inf

        os.makedirs(config.output_dir, exist_ok=True)

        results_dir = os.path.join(self.config.output_dir, "results")
        ckpts_dir = os.path.join(self.config.output_dir, "ckpts")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

        self.output_dir = config.output_dir
        self.ckpts_dir = ckpts_dir
        self.results_dir = results_dir

        # build optimizer & optionally SAM
        self.optimizer = self._build_optimizer(optimizer_config)
        if config.use_sam:
            base_opt = self.optimizer
            self.optimizer = SAM(base_opt, self.model, rho=config.sam_rho)

        self.scheduler = self._build_scheduler(config.scheduler) if config.scheduler else None
        self._setup_logging()

    def _setup_logging(self):
        # do not reconfigure root logger if already done
        self.logger = logging.getLogger("trainer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if self.config.wandb and _HAS_WANDB:
            wandb.init(project=self.config.wandb_project or "project", config=self.config.__dict__)
            self.wandb = wandb
        else:
            self.wandb = None

    def _build_optimizer(self, opt_cfg):
        opt_name = opt_cfg.get("name", self.config.optimizer).lower()
        lr = opt_cfg.get("lr")
        wd = opt_cfg.get("weight_decay")
        params = self.model.parameters()
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=opt_cfg.get("momentum", 0.9), weight_decay=wd)
        if opt_name == "adamax":
            return torch.optim.Adamax(params, lr=lr, weight_decay=wd)
        # fallback
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self, sched_cfg):
        if not sched_cfg:
            return None
        name = sched_cfg.get("name", "").lower()
        if name == "steplr":
            return torch.optim.lr_scheduler.StepLR(self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer,
                                                   step_size=sched_cfg.get("step_size", 10),
                                                   gamma=sched_cfg.get("gamma", 0.1))
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer,
                                                              T_max=sched_cfg.get("t_max", self.config.epochs))
        if name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer,
                                                       max_lr=sched_cfg.get("max_lr", self.config.lr),
                                                       steps_per_epoch=len(self.train_loader),
                                                       epochs=self.config.epochs)
        return None

    def save_checkpoint(self, epoch, step, is_best=False):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": (self.optimizer.base_optimizer.state_dict() if isinstance(self.optimizer, SAM) else self.optimizer.state_dict()),
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
                if isinstance(self.optimizer, SAM):
                    self.optimizer.base_optimizer.load_state_dict(opt_state)
                else:
                    self.optimizer.load_state_dict(opt_state)
        self.start_epoch = state.get("epoch", 0)
        self.global_step = state.get("step", 0)
        self.best_metric = state.get("best_metric", self.best_metric)
        self.logger.info(f"Loaded checkpoint from {path}")

    def infer(self, images):
        self.model.eval()
        return self.model(images)

    def train(self):
        self.model.train()

        # Log number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable parameters: {num_params:,}")

        cfg = self.config
        for epoch in range(self.start_epoch, cfg.epochs):
            epoch_start = time.time()

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="batch")
            for batch_idx, batch in enumerate(loop):
                self.global_step += 1
                loss = self._train_step(batch)
                if batch_idx % cfg.batch_log_interval == 0:
                    lr = self._current_lr()
                    self.logger.info(f"Epoch[{epoch}/{cfg.epochs}] Step[{batch_idx}] GlobalStep[{self.global_step}] Loss:{loss:.4f} LR:{lr:.6f}")
                    if self.wandb:
                        self.wandb.log({"train/loss": loss, "lr": lr, "step": self.global_step, "epoch": epoch})

                # eval by steps
                if cfg.eval_every_steps and self.global_step % cfg.eval_every_steps == 0:
                    metrics = self._evaluate(epoch, step=self.global_step)
                    self._maybe_checkpoint(metrics, epoch, self.global_step)

                if cfg.max_steps and self.global_step >= cfg.max_steps:
                    self.logger.info("Reached max steps. Stopping training.")
                    return

            # end epoch
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR) is False:
                self.scheduler.step()
            # eval by epochs
            if cfg.eval_every_epochs and (epoch + 1) % cfg.eval_every_epochs == 0:
                metrics = self._evaluate(epoch, step=self.global_step)
                self._maybe_checkpoint(metrics, epoch, self.global_step)

            elapsed = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1} done in {elapsed:.1f}s")

    def _train_step(self, batch):
        # expects batch to be dict with 'image' and 'label' at least
        images = batch.get("image")
        labels = batch.get("label")
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        logits = outputs["cls"] if isinstance(outputs, dict) and "cls" in outputs else outputs
        loss = self.criterion(logits, labels)

        if isinstance(self.optimizer, SAM):
            # SAM two-step
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            # second forward-backward
            outputs2 = self.model(images)
            logits2 = outputs2["cls"] if isinstance(outputs2, dict) and "cls" in outputs2 else outputs2
            loss2 = self.criterion(logits2, labels)
        
        else:
            # regular optimizer step
            loss.backward()
            self.optimizer.step()


        return loss.item()

    def _safe_compute_metrics(self, labels, probs):
        results = {}
        unique_classes = np.unique(labels)

        # Basic accuracy (always valid)
        preds = (probs >= 0.5).astype(int)
        results["acc"] = sk_metrics.accuracy_score(labels, preds)

        # Precision, recall, f1 (safe with one class → set zero_division=0)
        results["precision"] = sk_metrics.precision_score(labels, preds, zero_division=0)
        results["recall"] = sk_metrics.recall_score(labels, preds, zero_division=0)
        results["f1"] = sk_metrics.f1_score(labels, preds, zero_division=0)

        # AUC, AP, EER only if at least 2 classes
        if len(unique_classes) > 1:
            try:
                results["auc"] = sk_metrics.roc_auc_score(labels, probs)
            except Exception:
                results["auc"] = float("nan")

            try:
                results["ap"] = sk_metrics.average_precision_score(labels, probs)
            except Exception:
                results["ap"] = float("nan")

            # EER (custom)
            fpr, tpr, _ = sk_metrics.roc_curve(labels, probs)
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
            results["eer"] = eer
        else:
            results["auc"] = float("nan")
            results["ap"] = float("nan")
            results["eer"] = float("nan")

        return results


    def _evaluate(self, epoch, step):
        metrics = self._default_eval(epoch, step)
        self.logger.info(f"Eval at step {step}: {metrics}")
        if self.wandb:
            self.wandb.log({f"eval/{k}": v for k, v in metrics["global"].items()}, step=step)
        return metrics


    def _default_eval(self, epoch, step, name=None):
        self.model.eval()
        all_labels, all_probs, all_names = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                names = batch["dataset_name"]

                outputs = self.model(imgs)
                logits = outputs["cls"] if isinstance(outputs, dict) and "cls" in outputs else outputs
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

        # Global metrics
        results["global"] = self._safe_compute_metrics(labels, probs)

        # Per-dataset metrics
        for ds in np.unique(names):
            mask = names == ds
            if mask.sum() == 0:
                continue
            ds_labels, ds_probs = labels[mask], probs[mask]
            results[ds] = self._safe_compute_metrics(ds_labels, ds_probs)

        # Logging
        for ds, metrics in results.items():
            msg = f"[Eval - {ds}] " + " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
            self.logger.info(msg)
            if self.wandb:
                self.wandb.log({f"{ds}/{k}": v for k, v in metrics.items()}, step=self.global_step)

        # Save to JSON

        if epoch == -1:
            json_path = os.path.join(self.results_dir, f"metrics_{name}.json")
        else:
            json_path = os.path.join(self.results_dir, f"metrics_epoch{epoch}_step{step}.json")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"[Eval] Metrics saved to {json_path}")

        self.model.train()
        return results


    def _compute_metrics(self, labels, probs):
        from sklearn import metrics as _m

        auc = float(_m.roc_auc_score(labels, probs))
        fpr, tpr, _ = _m.roc_curve(labels, probs)
        fnr = 1 - tpr
        eer = float(fpr[np.argmin(np.abs(fnr - fpr))]) if len(fpr) > 0 else 0.0

        pred = (probs >= 0.5).astype(int)
        acc = float((pred == labels).mean())
        ap = float(_m.average_precision_score(labels, probs))

        precision = float(_m.precision_score(labels, pred, zero_division=0))
        recall = float(_m.recall_score(labels, pred, zero_division=0))
        f1 = float(_m.f1_score(labels, pred, zero_division=0))

        return {
            "auc": auc,
            "eer": eer,
            "acc": acc,
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _maybe_checkpoint(self, metrics: Dict[str, float], epoch: int, step: int):
        key = self.config.monitor_metric
        if key not in metrics["global"]:
            self.logger.warning(f"Monitor metric {key} not in eval metrics. Skipping best-checkpoint logic.")
            return
        val = metrics["global"][key]
        better = (val > self.best_metric) if self.config.monitor_mode == "max" else (val < self.best_metric)
        if better:
            self.best_metric = val
            self.save_checkpoint(epoch, step, is_best=True)

    def _current_lr(self):
        opt = (self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer)
        for g in opt.param_groups:
            return g.get("lr", 0.0)
