# training/base_trainer.py
import os
import math
import logging
import torch
from dataclasses import dataclass
from deepfense.training.optimizers.registry import OPTIMIZER_REGISTRY
from deepfense.training.schedulers.registry import SCHEDULER_REGISTRY

class BaseTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        os.makedirs(config.output_dir, exist_ok=True)
        self.logger = logging.getLogger("trainer")

        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -math.inf if config.monitor_mode == "max" else math.inf
        self.optimizer = None
        
    def save_checkpoint(self, state, is_best=False):
        ckpt_path = os.path.join(self.config.output_dir, "last.pth")
        torch.save(state, ckpt_path)
        if is_best:
            best_path = os.path.join(self.config.output_dir, "best.pth")
            torch.save(state, best_path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.global_step = state.get("global_step", 0)

    def train_step(self, batch):
        """Override in subclass."""
        raise NotImplementedError

    def evaluate(self):
        """Override in subclass."""
        raise NotImplementedError

    def _build_optimizer(self, opt_cfg):
        opt_name = opt_cfg.get("type", "adam").lower()
        optimizer_class = OPTIMIZER_REGISTRY[opt_name]
        params = self.model.parameters()
        return optimizer_class(params, opt_cfg)

    def _build_scheduler(self, sched_cfg):
        sched_name = sched_cfg.get("type", "").lower()
        opt = self.optimizer

        if sched_name is None or sched_name == "":
            return None
        
        scheduler_class = SCHEDULER_REGISTRY[sched_name]
        params = self.model.parameters()
        return scheduler_class(opt, sched_cfg)
