import os
import math
import logging
import torch
from dataclasses import dataclass
from deepfense.utils.registry import get_optimizer, get_scheduler


class BaseTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device

        os.makedirs(config.output_dir, exist_ok=True)
        self.logger = logging.getLogger("trainer")

        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -math.inf if getattr(config, "monitor_mode", "max") == "max" else math.inf
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
        # Expecting DictConfig or dict
        opt_name = opt_cfg.get("type", "adam") if isinstance(opt_cfg, dict) else opt_cfg.type
        opt_name = opt_name.lower()
        
        params = self.model.parameters()
        
        # Prepare config dict
        if hasattr(opt_cfg, "type"):
             # If it's a DictConfig, convert to dict
             from omegaconf import OmegaConf
             config = OmegaConf.to_container(opt_cfg, resolve=True)
        else:
             config = opt_cfg.copy()
             
        # Remove 'type' from config as it's not an optimizer param
        if "type" in config:
            del config["type"]

        from deepfense.utils.registry import build_optimizer
        return build_optimizer(opt_name, params, config)

    def _build_scheduler(self, sched_cfg):
        sched_name = sched_cfg.get("type", "") if isinstance(sched_cfg, dict) else sched_cfg.type
        sched_name = sched_name.lower()
        opt = self.optimizer

        if sched_name is None or sched_name == "":
            return None

        # Prepare config dict
        if hasattr(sched_cfg, "type"):
             from omegaconf import OmegaConf
             config = OmegaConf.to_container(sched_cfg, resolve=True)
        else:
             config = sched_cfg.copy()
             
        if "type" in config:
            del config["type"]

        from deepfense.utils.registry import build_scheduler
        return build_scheduler(sched_name, opt, config)
