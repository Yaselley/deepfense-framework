import torch
import pandas as pd
import numpy as np
import os
import logging

from deepfense.data.transforms.transforms import load_audio
from deepfense.data.base_dataset import BaseDataset
from deepfense.utils.registry import register_dataset, build_transforms_pipeline

logger = logging.getLogger(__name__)


@register_dataset("StandardDataset")
class StandardDataset(BaseDataset):
    """
    Dataset for audio deepfake detection.
    Handles reading Parquet metadata, mapping labels,
    applying transforms, and loading feature/audio files.
    
    Args:
        cfg (dict): Configuration with keys:
            - parquet_files (list[str]): Paths to parquet metadata files
            - label_map (dict): Mapping from label strings to integers
            - root_dir (str, optional): Base directory to prepend to paths in parquet
            - dataset_names (list[str], optional): Names for each parquet file
            - max_per_class (int, optional): Maximum samples per class
            - target_sr (int): Target sample rate (default: 16000)
            - mono (bool): Convert to mono (default: True)
            - base_transform (list): Base transform pipeline config
            - augment_transform (list): Augmentation pipeline config
    """

    def __init__(self, cfg):
        super().__init__()

        self.config_data = cfg
        self.label_map = self.config_data["label_map"]
        self.parquet_files = self.config_data["parquet_files"]
        self.dataset_names = self.config_data.get("dataset_names", None)
        
        self.root_dir = self.config_data.get("root_dir", None)

        self.max_per_class = self.config_data.get("max_per_class", None)

        self.base_transform_cfg = self.config_data.get("base_transform", None)
        self.augment_transform_cfg = self.config_data.get("augment_transform", None)

        self.base_transform = build_transforms_pipeline(self.base_transform_cfg)
        self.augment_transform = build_transforms_pipeline(
            self.augment_transform_cfg
        )

        # Load and concatenate Parquet metadata
        self.data = []
        for i, p_file in enumerate(self.parquet_files):
            # Check if parquet file exists
            if not os.path.exists(p_file):
                error_msg = (
                    f"Parquet file not found: {p_file}\n"
                    f"Please check the path in your configuration."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            df = pd.read_parquet(p_file)
            
            # Check if parquet file is empty
            if len(df) == 0:
                error_msg = (
                    f"Parquet file is empty: {p_file}\n"
                    f"Please ensure the file contains data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check if required columns exist
            required_columns = ["path", "label"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = (
                    f"Parquet file '{p_file}' is missing required columns: {missing_columns}\n"
                    f"Required columns: {required_columns}\n"
                    f"Available columns: {list(df.columns)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            df["dataset_name"] = (
                self.dataset_names[i] if i < len(self.dataset_names) else f"dataset_{i}"
            )
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        
        # Check if concatenated data is empty
        if len(self.data) == 0:
            error_msg = (
                f"No data found after loading parquet files: {self.parquet_files}\n"
                f"Please check your data configuration."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Map labels
        self.data["label"] = self.data["label"].map(self.label_map)

        # Optionally limit samples per class
        if self.max_per_class is not None:
            limited_data = []
            for label in self.data["label"].unique():
                df_label = self.data[self.data["label"] == label]
                limited_data.append(
                    df_label.sample(n=min(self.max_per_class, len(df_label)))
                )
            self.data = pd.concat(limited_data, ignore_index=True)

    def _get_audio_path(self, path):
        """Get full audio path, prepending root_dir if specified."""
        if self.root_dir is not None:
            return os.path.join(self.root_dir, path[1:])
        return path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        audio_path = self._get_audio_path(row["path"])

        # Check if audio file exists
        if not os.path.exists(audio_path):
            error_msg = (
                f"Audio file not found: {audio_path}\n"
                f"Row ID: {row.get('ID', idx)}\n"
                f"Dataset: {row.get('dataset_name', 'unknown')}\n"
                f"Original path in parquet: {row['path']}\n"
                f"Root directory: {self.root_dir}\n"
                f"Please check that the file exists and the path is correct."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        x = load_audio(
            path=audio_path,
            target_sr=self.config_data.get("target_sr", 16000),
            mono=self.config_data.get("mono", True),
        )

        if self.base_transform:
            x = self.base_transform(x)
        if self.augment_transform:
            x = self.augment_transform(x)

        return {
            "ID": row.get("ID", f"{row['dataset_name']}_{idx}"),
            "x": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "dataset_name": row["dataset_name"],
        }
