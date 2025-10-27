import os
import argparse
import yaml
import logging
from datetime import datetime

from deepfense.data.data_utils import build_dataloader
from deepfense.models.registry import DETECTOR
from deepfense.training.registry import TRAINER_REGISTRY, TRAINER_CONFIG_REGISTRY
from deepfense.training.losses.registry import LOSS_REGISTRY

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(output_dir, exp_name, cfg):
    """Setup structured logging and clean folder creation."""
    # Ensure clean and unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Paths
    log_file = os.path.join(exp_dir, "train.log")
    config_out = os.path.join(exp_dir, "config.yaml")

    # Configure logging (console + file)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                # Console
            logging.FileHandler(log_file, mode="w") # File
        ]
    )

    # Save a copy of the configuration for reproducibility
    with open(config_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Configuration saved to: {config_out}")
    logging.info("Logging initialized successfully.\n")

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
    output_dir = setup_logging(base_output_dir, exp_name, cfg)
    cfg["trainer"]["params"]["output_dir"] = output_dir  # override with exp-specific path


    # Logging
    logger = logging.getLogger("train")
    logger.info(f"Experiment directory: {output_dir}")

    # DataLoaders
    # add the labels to the config for dataset initialization
    cfg["data"]["train"]["label_map"] = cfg["data"]["label_map"]
    cfg["data"]["val"]["label_map"] = cfg["data"]["label_map"]
    
    # add sampling_rate to the config for dataset initialization
    cfg["data"]["train"]["sampling_rate"] = cfg["data"]["sampling_rate"]
    cfg["data"]["val"]["sampling_rate"] = cfg["data"]["sampling_rate"]

    train_loader = build_dataloader(cfg["data"]["train"])
    val_loader = build_dataloader(cfg["data"]["val"])


    # Detector (frontend + backend)
    detector_cfg = cfg["detector"]
    detector = DETECTOR[detector_cfg["type"]](detector_cfg)

    # Loss
    loss_name = cfg["loss"]["type"]
    loss_params = cfg["loss"].get("params", {})
    criterion = LOSS_REGISTRY[loss_name](**loss_params)

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
        criterion=criterion,
        optimizer_config=cfg.get("optimizer_config"),
        config=TrainerConfig,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
