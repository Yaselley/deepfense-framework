import torch
import logging
from torch.utils.data import DataLoader, DistributedSampler

import deepfense.data  # noqa: F401 — register StandardDataset, TemporalSegmentationDataset, etc.
from deepfense.utils.registry import build_dataset

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Collate a batch of dicts {"ID", "x", "label", "dataset_name"}.

    Pads clips to the longest in the batch and emits an audio ``mask``
    (1=real sample, 0=zero-padding). If items contain ``frame_labels``,
    pads them to a common frame length and adds ``frame_mask``.
    """
    xs = [item["x"] for item in batch]
    labels = [item["label"] for item in batch]
    dataset_names = [item["dataset_name"] for item in batch]
    ids = [item["ID"] for item in batch]
    has_frames = "frame_labels" in batch[0]

    frame_label_list = [item["frame_labels"] for item in batch] if has_frames else None

    if xs[0].ndim == 1:
        max_len = max(x.shape[0] for x in xs)
    else:
        max_len = max(x.shape[-1] for x in xs)

    padded_xs = []
    masks = []

    for x in xs:
        if x.ndim == 1:
            seq_len = x.shape[0]
            if seq_len < max_len:
                x = torch.cat([x, torch.zeros(max_len - seq_len, dtype=x.dtype)])
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.float32),
                    torch.zeros(max_len - seq_len, dtype=torch.float32),
                ])
            else:
                mask = torch.ones(max_len, dtype=torch.float32)
        else:
            n_aug, seq_len = x.shape[0], x.shape[1]
            if seq_len < max_len:
                pad = torch.zeros((n_aug, max_len - seq_len), dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.float32),
                    torch.zeros(max_len - seq_len, dtype=torch.float32),
                ])
            else:
                mask = torch.ones(max_len, dtype=torch.float32)

        padded_xs.append(x)
        masks.append(mask)

    x = torch.stack(padded_xs, dim=0)
    mask = torch.stack(masks, dim=0)
    label = torch.stack(labels, dim=0)

    out = {
        "x": x,
        "label": label,
        "dataset_name": dataset_names,
        "mask": mask,
        "ID": ids,
    }

    if has_frames and frame_label_list is not None:
        max_f = max(fl.shape[0] for fl in frame_label_list)
        padded_fl = []
        f_masks = []
        for fl in frame_label_list:
            nf = fl.shape[0]
            if nf < max_f:
                fl = torch.cat([fl, torch.full((max_f - nf,), -100, dtype=torch.long)])
                fm = torch.cat([
                    torch.ones(nf, dtype=torch.float32),
                    torch.zeros(max_f - nf, dtype=torch.float32),
                ])
            elif nf > max_f:
                fl = fl[:max_f]
                fm = torch.ones(max_f, dtype=torch.float32)
            else:
                fm = torch.ones(max_f, dtype=torch.float32)
            padded_fl.append(fl)
            f_masks.append(fm)
        out["frame_labels"] = torch.stack(padded_fl, dim=0)
        out["frame_mask"] = torch.stack(f_masks, dim=0)

    return out


def build_dataloader(config):
    """Build a DataLoader given a dataset name and configuration."""
    dataset_name = config["dataset_type"]
    ds = build_dataset(dataset_name, cfg=config)

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

    sampler = None
    if config.get("distributed", False):
        sampler = DistributedSampler(
            ds,
            num_replicas=config.get("world_size", 1),
            rank=config.get("rank", 0),
            shuffle=shuffle,
        )
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
