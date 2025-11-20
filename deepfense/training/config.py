from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from deepfense.training.registry import register_config_trainer


@register_config_trainer("StandardConfig")
@dataclass
class StandardConfig:
    output_dir: str
    device: str = "cuda"
    epochs: int = 10
    max_steps: Optional[int] = None
    eval_every_steps: Optional[int] = None
    eval_every_epochs: Optional[int] = 1
    save_every_epochs: Optional[int] = 1
    batch_log_interval: Optional[int] = None
    seed: Optional[int] = None
    use_amp: bool = False
    wandb: bool = False
    wandb_project: Optional[str] = None
    optimizer: str = "adam"
    scheduler: Optional[dict] = None
    use_sam: bool = False
    sam_rho: float = 0.05
    monitor_metric: str = "loss"  # metric name to select best checkpoint
    monitor_mode: str = "min"  # 'max' or 'min'
    batch_size: int = 32
    num_workers: int = 4
