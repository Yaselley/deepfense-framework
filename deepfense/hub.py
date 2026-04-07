"""Download DeepFense datasets and pretrained models from HuggingFace."""

import os
import logging
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download, list_repo_files, HfApi

logger = logging.getLogger(__name__)

HF_ORG = "DeepFense"

DATASETS = [
    "ASVSpoof19",
    "CompSpoof",
    "CtrSVDD",
    "DECRO",
    "EnvSDD",
    "FakeMusicCaps",
    "ODSS",
    "SONICS",
    "SpeechFake",
    "SpoofCeleb",
    "VCapAV",
    "WaveFake",
]


def list_datasets():
    """Return the list of available dataset names on HuggingFace."""
    return list(DATASETS)


def list_models(pattern: Optional[str] = None) -> list[str]:
    """List available pretrained models on HuggingFace.

    Args:
        pattern: Optional substring filter (e.g. "WavLM", "ASV19", "Nes2Net").
    """
    api = HfApi()
    models = api.list_models(author=HF_ORG)
    names = sorted(m.id.split("/", 1)[1] for m in models)
    if pattern:
        pattern_lower = pattern.lower()
        names = [n for n in names if pattern_lower in n.lower()]
    return names


def download_dataset(
    name: str,
    output_dir: Union[str, Path] = "./data",
) -> list[str]:
    """Download a DeepFense dataset (parquet files) from HuggingFace.

    Args:
        name: Dataset name (e.g. "ASVSpoof19", "CompSpoof").
        output_dir: Local directory to save parquet files.

    Returns:
        List of local paths to downloaded parquet files.
    """
    repo_id = f"{HF_ORG}/{name}"
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not parquet_files:
        raise ValueError(
            f"No parquet files found in dataset '{repo_id}'. "
            f"Available files: {files}"
        )

    downloaded = []
    for pf in parquet_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        downloaded.append(os.path.abspath(local_path))
        logger.info(f"Downloaded {pf} -> {local_path}")

    print(f"Downloaded {len(downloaded)} parquet file(s) to {output_dir}/")
    for p in downloaded:
        print(f"  {p}")

    return downloaded


def download_model(
    name: str,
    output_dir: Union[str, Path] = "./models",
) -> dict[str, str]:
    """Download a pretrained DeepFense model from HuggingFace.

    Args:
        name: Model name (e.g. "ASV19_WavLM_Nes2Net_NoAug_Seed42").
            Can be just the model name or full "DeepFense/model_name".
        output_dir: Local directory to save model files.

    Returns:
        Dict with keys "checkpoint" and "config" pointing to local paths.
    """
    if "/" not in name:
        repo_id = f"{HF_ORG}/{name}"
    else:
        repo_id = name

    model_name = repo_id.split("/", 1)[1]
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="best_model.pth",
        local_dir=str(output_dir),
    )

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.yaml",
        local_dir=str(output_dir),
    )

    result = {
        "checkpoint": os.path.abspath(ckpt_path),
        "config": os.path.abspath(config_path),
    }

    print(f"Downloaded model to {output_dir}/")
    print(f"  Checkpoint: {result['checkpoint']}")
    print(f"  Config:     {result['config']}")

    return result
