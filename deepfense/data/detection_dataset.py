import torch
import pandas as pd
import numpy as np

from deepfense.data.transforms.transforms import load_audio
from deepfense.data.base_dataset import BaseDataset
from deepfense.data.registry import register_dataset
from deepfense.data.transforms.registry import build_transforms_from_config

@register_dataset("StandardDataset")
class StandardDataset(BaseDataset):
    """
    Dataset for audio deepfake detection.
    Handles reading Parquet metadata, mapping labels,
    applying transforms, and loading feature/audio files.
    """

    def __init__(self, cfg):
        super().__init__()

        self.config_data = cfg
        self.label_map = self.config_data["label_map"]
        self.parquet_files = self.config_data["parquet_files"]
        self.dataset_names = self.config_data.get("dataset_names", None)

        self.max_per_class = self.config_data.get("max_per_class", None)

        self.base_transform_cfg = self.config_data.get("base_transform", None)
        self.augment_transform_cfg = self.config_data.get("augment_transform", None)

        print(self.base_transform_cfg)
        self.base_transform = build_transforms_from_config(self.base_transform_cfg)
        self.augment_transform = build_transforms_from_config(self.augment_transform_cfg)

        # Load and concatenate Parquet metadata
        self.data = []
        for i, p_file in enumerate(self.parquet_files):
            df = pd.read_parquet(p_file)
            df["dataset_name"] = self.dataset_names[i] if i < len(self.dataset_names) else f"dataset_{i}"
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load audio
        x = load_audio(
            path=row["path"],
            target_sr=self.config_data.get("target_sr", 16000),
            mono=self.config_data.get("mono", True)
        )

        if self.base_transform:
            x = self.base_transform(x)
        if self.augment_transform:
            x = self.augment_transform(x)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "dataset_name": row["dataset_name"],
        }

        