import torch
import logging
from torch.utils.data import DataLoader
from deepfense.utils.registry import build_dataset

logger = logging.getLogger(__name__)


def collate_fn(batch, max_pad=None):
    """
    Collate a batch of dicts {"ID", "x", "label", "dataset_name"}.

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
            pad_shape = (max_len - seq_len, *x.shape[1:])
            x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)
            mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.float32),
                    torch.zeros(max_len - seq_len, dtype=torch.float32),
                ]
            )
        else:
            if max_pad and seq_len > max_pad:
                 x = x[:max_pad]
                 mask = torch.ones(max_pad, dtype=torch.float32)
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
        "ID": ids,
    }


def build_dataloader(config):
    """
    Builds a DataLoader given a dataset name and configuration.
    """

    dataset_name = config["dataset_type"]
    # Build dataset using registry
    ds = build_dataset(dataset_name, cfg=config)

    # Check if dataset is empty
    if len(ds) == 0:
        error_msg = (
            f"Dataset '{dataset_name}' is empty. Please check your data configuration:\n"
            f"  - parquet_files: {config.get('parquet_files', 'not specified')}\n"
            f"  - root_dir: {config.get('root_dir', 'not specified')}\n"
            f"  - label_map: {config.get('label_map', 'not specified')}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    batch_size = config.get("batch_size", 8)
    shuffle = config.get("shuffle", False)
    num_workers = config.get("num_workers", 0)

    # Return DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, max_pad=config.get("max_len", None)),
        num_workers=num_workers
    )
