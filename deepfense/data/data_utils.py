import torch
from torch.utils.data import DataLoader
from deepfense.data.registry import get_dataset_class


def collate_fn(batch, max_pad=None, pad_strategy="zero"):
    """
    Collate a batch of dicts {"x", "label", "dataset_name"}.

    Returns a dict:
        {
            "x": Tensor of shape (B, max_len, ...),
            "label": Tensor of shape (B,),
            "dataset_name": list[str],
            "mask": Tensor of shape (B, max_len), 1=valid, 0=pad
            "ID": list[str]
        }
    """
    xs = [item["x"] for item in batch]
    labels = [item["label"] for item in batch]
    dataset_names = [item["dataset_name"] for item in batch]
    ids = [item["ID"] for item in batch]

    # Determine max length
    max_len = max(x.shape[0] for x in xs)
    if max_pad is not None:
        max_len = max(max_len, max_pad)

    padded_xs = []
    masks = []

    for x in xs:
        seq_len = x.shape[0]

        if seq_len < max_len:
            if pad_strategy == "replicate":
                repeat_times = (max_len + seq_len - 1) // seq_len
                x_repeated = x.repeat(repeat_times, *([1] * (x.dim() - 1)))
                x = x_repeated[:max_len]
                mask = torch.ones(max_len, dtype=torch.float32)
            elif pad_strategy == "zero":
                pad_shape = (max_len - seq_len, *x.shape[1:])
                x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)
                mask = torch.cat([torch.ones(seq_len, dtype=torch.float32),
                                  torch.zeros(max_len - seq_len, dtype=torch.float32)])
            else:
                raise ValueError(f"Unknown pad_strategy: {pad_strategy}")
        else:
            mask = torch.ones(max_len, dtype=torch.float32)

        padded_xs.append(x)
        masks.append(mask)

    x = torch.stack(padded_xs, dim=0)
    mask = torch.stack(masks, dim=0)
    label = torch.stack(labels, dim=0)

    return {
        "x": x,
        "label": label,
        "dataset_name": dataset_names,
        "mask": mask,
        "ID": ids
    }



def build_dataloader(config, split="train"):
    """
    Builds a DataLoader given a dataset name and configuration.
    Example expected config:
        config = {
            "data": {
                "dataset_type": "DetectionDataset",
                "train": {
                    "parquet_files": [...],
                    "names": [...],
                },
                "label_map": {...}
            }
            "base_transform": {...},
            "augment_transform": {...},
        }
    """

    dataset_name = config["dataset_type"]
    # Fetch dataset class from registry
    DatasetClass = get_dataset_class(dataset_name)
    
    batch_size = config.get("batch_size", 8)
    shuffle = True if split == "train" else False
    pad_strategy = config.get("pad_strategy", "zero")

    # Initialize dataset
    ds = DatasetClass(
        cfg=config
    )

    # Return DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_strategy=pad_strategy),
    )
