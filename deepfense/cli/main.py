"""Main CLI entry point for DeepFense."""
import click

from deepfense.cli.commands.train import train
from deepfense.cli.commands.test import test
from deepfense.cli.commands.list_components import list_components
from deepfense.cli.commands.download import download


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """
    DeepFense -- Modular Deepfake Audio Detection

    \b
    Commands:
      train      Train a model from a YAML config
      test       Evaluate a trained model
      list       Show all registered components
      download   Download datasets/models from HuggingFace
    """
    pass


# Register all subcommands
cli.add_command(train)
cli.add_command(test)
cli.add_command(list_components, name='list')
cli.add_command(download)


if __name__ == "__main__":
    cli()

