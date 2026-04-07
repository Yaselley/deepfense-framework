import os
import argparse
import logging
from datetime import datetime
from omegaconf import OmegaConf

import torch
import torch.distributed as dist

from deepfense.training.set_seed import set_seed
from deepfense.data.data_utils import build_dataloader
from deepfense.models import *
from deepfense.utils.registry import build_detector, build_trainer


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched via torchrun / torch.distributed.launch."""
    if "RANK" not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return True


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def load_config(config_path):
    return OmegaConf.load(config_path)


def setup_logging(output_dir, exp_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "train.log")

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("train")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Logging re-configured successfully. All logs saving to {log_file}\n")

    return exp_dir


def validate_config(cfg):
    logger = logging.getLogger("train")
    for split in ["train", "val"]:
        if split in cfg.data:
            ds_cfg = cfg.data[split]
            p_files = ds_cfg.get("parquet_files", [])
            ds_names = ds_cfg.get("dataset_names", [])

            if not p_files:
                logger.error(f"No parquet files specified for {split}!")
                raise ValueError(f"Missing parquet_files in data.{split}")

            if ds_names and len(p_files) != len(ds_names):
                logger.warning(f"Mismatch in {split}: {len(p_files)} files vs {len(ds_names)} names.")

            for f in p_files:
                if not os.path.exists(f):
                    logger.error(f"Parquet file not found: {f}")
                    raise FileNotFoundError(f"{f} does not exist")

    if not cfg.model.get("loss"):
        logger.warning("No loss function defined in model config!")

    logger.info("Configuration validation passed.")


class _DDPForwardWrapper(torch.nn.Module):
    """Wraps a detector so forward() includes loss computation,
    ensuring all parameters (including loss modules) participate
    in a single DDP forward pass."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, x, mask=None, labels=None):
        outputs = self.detector(x, mask=mask) if mask is not None else self.detector(x)
        if labels is not None:
            outputs["loss"] = self.detector.compute_loss(outputs, labels)
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    ddp = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    cfg = load_config(args.config)

    if is_main_process():
        validate_config(cfg)

    base_output_dir = cfg.get("output_dir", "./outputs")
    exp_name = cfg.get("exp_name", "default_exp")

    if is_main_process():
        output_dir = setup_logging(base_output_dir, exp_name)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"{exp_name}_{timestamp}")

    if ddp:
        output_dir_list = [output_dir]
        dist.broadcast_object_list(output_dir_list, src=0)
        output_dir = output_dir_list[0]

    cfg.training.output_dir = output_dir

    logger = logging.getLogger("train")

    if is_main_process():
        config_out = os.path.join(output_dir, "config.yaml")
        try:
            OmegaConf.save(cfg, config_out)
            logger.info(f"Final configuration saved to: {config_out}")
        except Exception as e:
            logger.error(f"Failed to save final config: {e}")

    set_seed(cfg.get("seed", 42) + rank)

    if "label_map" in cfg.data:
        cfg.data.train.label_map = cfg.data.label_map
        cfg.data.val.label_map = cfg.data.label_map
        if "test" in cfg.data:
            cfg.data.test.label_map = cfg.data.label_map

    if "sampling_rate" in cfg.data:
        cfg.data.train.sampling_rate = cfg.data.sampling_rate
        cfg.data.val.sampling_rate = cfg.data.sampling_rate
        if "test" in cfg.data:
            cfg.data.test.sampling_rate = cfg.data.sampling_rate

    train_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    val_cfg = OmegaConf.to_container(cfg.data.val, resolve=True)

    if ddp:
        train_cfg["distributed"] = True
        train_cfg["rank"] = rank
        train_cfg["world_size"] = world_size

    train_loader = build_dataloader(train_cfg)
    val_loader = build_dataloader(val_cfg)

    label_map = cfg.data.get("label_map", {"bonafide": 1, "spoof": 0})
    if hasattr(label_map, "get"):
        bonafide_label = label_map.get("bonafide", 1)
    else:
        bonafide_label = getattr(label_map, "bonafide", 1)

    cfg.model.bonafide_label = bonafide_label

    if ddp:
        device = f"cuda:{rank}"
    else:
        device = cfg.training.get("device", "cuda")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    detector = build_detector(cfg.model.type, model_cfg)
    detector.to(device)

    if ddp:
        detector = _DDPForwardWrapper(detector)
        detector = torch.nn.parallel.DistributedDataParallel(
            detector, device_ids=[rank], find_unused_parameters=True,
        )
        if is_main_process():
            logger.info(f"DDP enabled: {world_size} GPUs")

    cfg.training.device = device
    cfg.training._ddp = ddp
    cfg.training._rank = rank
    cfg.training._world_size = world_size

    trainer_type = cfg.training.get("trainer", "StandardTrainer")

    trainer = build_trainer(
        trainer_type,
        model=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.training,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    cleanup_distributed()


if __name__ == "__main__":
    main()
