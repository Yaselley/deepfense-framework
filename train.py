import os
import argparse
import yaml
import logging
from datetime import datetime

from deepfense.data.data_utils import build_dataloader
from deepfense.models.registry import DETECTOR
from deepfense.training.registry import TRAINER_REGISTRY, TRAINER_CONFIG_REGISTRY

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(output_dir, exp_name):
    """Setup structured logging and clean folder creation."""
    # Ensure clean and unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Paths
    log_file = os.path.join(exp_dir, "train.log")
    # config_out path is no longer needed here
    
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
    
    logger = logging.getLogger("train") # Matches what you use in main()
    
    logger.info(f"Experiment directory: {exp_dir}")
    # Removed config log message
    logger.info(f"Logging re-configured successfully. All logs saving to {log_file}\n")

    return exp_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup experiment directory + logging
    base_output_dir = cfg["output_dir"]
    exp_name = cfg.get("exp_name", "default_exp")
    
    # --- MODIFIED: Call new setup_logging ---
    output_dir = setup_logging(base_output_dir, exp_name)

    # Logging
    logger = logging.getLogger("train")
    logger.info(f"Experiment directory: {output_dir}")

    # --- MODIFICATIONS: Apply all config overrides ---
    cfg["trainer"]["params"]["output_dir"] = output_dir  # override with exp-specific path

    # add the labels to the config for dataset initialization
    cfg["data"]["train"]["label_map"] = cfg["data"]["label_map"]
    cfg["data"]["val"]["label_map"] = cfg["data"]["label_map"]
    
    # add sampling_rate to the config for dataset initialization
    cfg["data"]["train"]["sampling_rate"] = cfg["data"]["sampling_rate"]
    cfg["data"]["val"]["sampling_rate"] = cfg["data"]["sampling_rate"]

    # --- NEW: Save the *final* modified config ---
    config_out = os.path.join(output_dir, "config.yaml")
    try:
        with open(config_out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info(f"Final configuration saved to: {config_out}")
    except Exception as e:
        logger.error(f"Failed to save final config: {e}")
    # --- END NEW SECTION ---

    # DataLoaders
    train_loader = build_dataloader(cfg["data"]["train"])
    val_loader = build_dataloader(cfg["data"]["val"])

    # Detector (frontend + backend + lossMapper)
    detector_cfg = cfg["detector"]
    loss_cfg = cfg["loss"]
    detector_cfg["loss"] = loss_cfg

    detector = DETECTOR[detector_cfg["type"]](detector_cfg)

    # Trainer
    trainer_type = cfg["trainer"]["type"]
    trainer_config_type = cfg["trainer"]["config_type"]
    trainer_params = cfg["trainer"]["params"]

    TrainerConfig = TRAINER_CONFIG_REGISTRY[trainer_config_type](**trainer_params)
    TrainerClass = TRAINER_REGISTRY[trainer_type]

    trainer = TrainerClass(
        model=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_config=cfg.get("optimizer_config"),
        scheduler_config=cfg.get("scheduler_config", None),
        metrics_config=cfg.get("metrics", None),
        config=TrainerConfig,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()