"""Download datasets and models from HuggingFace."""
import click


@click.group()
def download():
    """Download datasets and pretrained models from HuggingFace."""
    pass


@download.command("dataset")
@click.argument("name")
@click.option("--output-dir", "-o", default="./data", help="Directory to save parquet files")
def download_dataset(name, output_dir):
    """Download a dataset by name.

    \b
    Examples:
        deepfense download dataset ASVSpoof19
        deepfense download dataset CompSpoof --output-dir ./my_data
    """
    from deepfense.hub import download_dataset as _download
    _download(name, output_dir=output_dir)


@download.command("model")
@click.argument("name")
@click.option("--output-dir", "-o", default="./models", help="Directory to save model files")
def download_model(name, output_dir):
    """Download a pretrained model by name.

    \b
    Examples:
        deepfense download model ASV19_WavLM_Nes2Net_NoAug_Seed42
        deepfense download model CodecFake_EAT_Nes2Net_NoAug_Seed240
    """
    from deepfense.hub import download_model as _download
    _download(name, output_dir=output_dir)


@download.command("list-datasets")
def list_datasets_cmd():
    """List all available datasets on HuggingFace."""
    from deepfense.hub import list_datasets
    click.echo("\nAvailable datasets (https://huggingface.co/DeepFense):\n")
    for name in list_datasets():
        click.echo(f"  {name}")
    click.echo()


@download.command("list-models")
@click.option("--filter", "-f", "pattern", default=None, help="Filter by substring (e.g. 'WavLM', 'ASV19')")
@click.option("--limit", "-n", default=20, help="Max number of results to show")
def list_models_cmd(pattern, limit):
    """List available pretrained models on HuggingFace.

    \b
    Examples:
        deepfense download list-models
        deepfense download list-models --filter WavLM
        deepfense download list-models --filter ASV19 --limit 50
    """
    from deepfense.hub import list_models
    models = list_models(pattern=pattern)
    total = len(models)

    click.echo(f"\nPretrained models on HuggingFace ({total} found):\n")
    for name in models[:limit]:
        click.echo(f"  DeepFense/{name}")
    if total > limit:
        click.echo(f"\n  ... and {total - limit} more. Use --limit to show more.")
    click.echo()
