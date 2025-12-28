"""Main CLI entry point for DeepFense."""
import click

from deepfense.cli.commands.train import train
from deepfense.cli.commands.test import test
from deepfense.cli.commands.list_components import list_components


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    DeepFense CLI - A Modular Framework for Deepfake Audio Detection
    
    Use subcommands to train models, test models, list components, and generate data.
    """
    pass


# Register all subcommands
cli.add_command(train)
cli.add_command(test)
cli.add_command(list_components, name='list')


if __name__ == "__main__":
    cli()

