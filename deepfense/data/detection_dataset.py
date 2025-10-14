import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from data.transforms import build_transforms_from_config
from data.registry import AUGMENTATION_REGISTRY

class DetectionDataset(Dataset):
    def __init__(self, parquet_files, names, root_dir, always_transform_config=None,
                 optional_transform_config=None, label_map=None, max_per_class=None,
                 seed=42):
        np.random.seed(seed)
        self.label_map = label_map
        self.always_transform = build_transforms_from_config(always_transform_config) \
            if always_transform_config else None
        self.optional_transform = build_transforms_from_config(optional_transform_config) \
            if optional_transform_config else None

        self.data = []
        for i, p_file in enumerate(parquet_files):
            df = pd.read_parquet(p_file)
            df["dataset_name"] = names[i] if i < len(names) else f"dataset_{i}"
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        self.data["label"] = self.data["label"].map(label_map).fillna(-1).astype(int)

        if max_per_class is not None:
            limited_data = []
            for label in self.data["label"].unique():
                df_label = self.data[self.data["label"] == label]
                limited_data.append(df_label.sample(n=min(max_per_class, len(df_label)), random_state=seed))
            self.data = pd.concat(limited_data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = np.load(f"{row['dataset_name']}/{row['file_path']}")  # placeholder for actual loading
        if self.always_transform:
            x = self.always_transform(x)
        if self.optional_transform:
            x = self.optional_transform(x)
        return torch.tensor(x, dtype=torch.float32), row["label"]

def collate_fn(batch, dynamic_batch=False):
    xs, ys = zip(*batch)
    if dynamic_batch:
        max_len = max([x.shape[0] for x in xs])
        xs = [torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in xs]
    xs = torch.stack(xs)
    ys = torch.tensor(ys)
    return xs, ys

def build_dataloader(config, split="train", batch_size=8, shuffle=True,
                     dynamic_batch=False, seed=42):
    dataset_cfg = config["data"][split]
    ds = DetectionDataset(
        parquet_files=dataset_cfg["parquet_files"],
        names=dataset_cfg.get("names"),
        root_dir=dataset_cfg.get("root_dir"),
        always_transform_config=dataset_cfg.get("always_transform"),
        optional_transform_config=dataset_cfg.get("optional_transform"),
        label_map=config["data"]["label_map"],
        seed=seed
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: collate_fn(b, dynamic_batch))
