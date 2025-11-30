import os
import argparse
import yaml
import logging
from datetime import datetime
from omegaconf import OmegaConf

from deepfense.training.set_seed import set_seed
from deepfense.data.data_utils import build_dataloader
from deepfense.models import * # Import models to register them
from deepfense.utils.registry import build_detector, build_trainer


def load_config(config_path):
    # Load as OmegaConf for better dot access and type safety later
    return OmegaConf.load(config_path)


def setup_logging(output_dir, exp_name):
    """Setup structured logging and clean folder creation."""
    # Ensure clean and unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Paths
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
    """
    Perform basic validation on the configuration to catch errors early.
    """
    logger = logging.getLogger("train")
    
    # 1. Check Data Config
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

            # Check file existence
            for f in p_files:
                if not os.path.exists(f):
                    logger.error(f"Parquet file not found: {f}")
                    raise FileNotFoundError(f"{f} does not exist")

    # 2. Check Model Config
    if not cfg.model.get("loss"):
        logger.warning("No loss function defined in model config!")

    logger.info("Configuration validation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    
    # Validate config
    validate_config(cfg)

    # Setup experiment directory + logging
    base_output_dir = cfg.get("output_dir", "./outputs")
    exp_name = cfg.get("exp_name", "default_exp")

    output_dir = setup_logging(base_output_dir, exp_name)

    # Update config with actual output_dir
    cfg.training.output_dir = output_dir
    
    # Logging
    logger = logging.getLogger("train")
    
    # Save the final configuration
    config_out = os.path.join(output_dir, "config.yaml")
    try:
        OmegaConf.save(cfg, config_out)
        logger.info(f"Final configuration saved to: {config_out}")
    except Exception as e:
        logger.error(f"Failed to save final config: {e}")

    # set seed
    set_seed(cfg.get("seed", 42))

    # DataLoaders
    # Propagate global settings to data configs if needed
    # e.g. sampling_rate, label_map
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

    train_loader = build_dataloader(OmegaConf.to_container(cfg.data.train, resolve=True))
    val_loader = build_dataloader(OmegaConf.to_container(cfg.data.val, resolve=True))

    # Detector
    # Expecting model config to contain frontend, backend, loss
    
    # Inject bonafide_label into model config so losses know about it
    # Usually label_map is {"bonafide": 1, "spoof": 0}
    label_map = cfg.data.get("label_map", {"bonafide": 1, "spoof": 0})
    # Handle both OmegaConf and dict
    if hasattr(label_map, "get"):
        bonafide_label = label_map.get("bonafide", 1)
    else:
        # Assuming dict-like access works or attribute access
        bonafide_label = getattr(label_map, "bonafide", 1)
    
    cfg.model.bonafide_label = bonafide_label

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    detector = build_detector(cfg.model.type, model_cfg)
    detector.to(cfg.training.get("device", "cuda"))

    # Trainer
    trainer_type = cfg.training.get("trainer", "StandardTrainer")
    
    # We pass the 'training' config section to the trainer
    trainer = build_trainer(
        trainer_type,
        model=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.training,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
