"""List all available components in DeepFense."""
import click

# Import models and other components to register them
from deepfense.models import *
from deepfense.data import *
from deepfense.training import *

from deepfense.utils.registry import (
    FRONTEND_REGISTRY,
    BACKEND_REGISTRY,
    LOSS_REGISTRY,
    DATASET_REGISTRY,
    TRANSFORM_REGISTRY,
    OPTIMIZER_REGISTRY,
    TRAINER_REGISTRY,
)


@click.command()
@click.option("--component-type", "-t", type=click.Choice([
    "all", "frontends", "backends", "losses", "datasets", 
    "augmentations", "optimizers", "trainers"
], case_sensitive=False), default="all", help="Type of component to list")
def list_components(component_type):
    """
    List all available components in DeepFense.
    
    Example:
    
        deepfense list
        
        deepfense list --component-type backends
    """
    component_type = component_type.lower()
    
    if component_type == "all" or component_type == "frontends":
        frontends = FRONTEND_REGISTRY.list()
        if frontends:
            click.echo("\nFrontends:")
            for name in sorted(frontends):
                click.echo(f"  • {name}")
        elif component_type == "frontends":
            click.echo("  (no frontends registered)")
    
    if component_type == "all" or component_type == "backends":
        backends = BACKEND_REGISTRY.list()
        if backends:
            click.echo("\nBackends:")
            for name in sorted(backends):
                click.echo(f"  • {name}")
        elif component_type == "backends":
            click.echo("  (no backends registered)")
    
    if component_type == "all" or component_type == "losses":
        losses = LOSS_REGISTRY.list()
        if losses:
            click.echo("\nLosses:")
            for name in sorted(losses):
                click.echo(f"  • {name}")
        elif component_type == "losses":
            click.echo("  (no losses registered)")
    
    if component_type == "all" or component_type == "datasets":
        datasets = DATASET_REGISTRY.list()
        if datasets:
            click.echo("\nDatasets:")
            for name in sorted(datasets):
                click.echo(f"  • {name}")
        elif component_type == "datasets":
            click.echo("  (no datasets registered)")
    
    if component_type == "all" or component_type == "augmentations":
        augmentations = TRANSFORM_REGISTRY.list()
        if augmentations:
            click.echo("\nAugmentations:")
            for name in sorted(augmentations):
                click.echo(f"  • {name}")
        elif component_type == "augmentations":
            click.echo("  (no augmentations registered)")
    
    if component_type == "all" or component_type == "optimizers":
        optimizers = OPTIMIZER_REGISTRY.list()
        if optimizers:
            click.echo("\nOptimizers:")
            for name in sorted(optimizers):
                click.echo(f"  • {name}")
        elif component_type == "optimizers":
            click.echo("  (no optimizers registered)")
    
    if component_type == "all" or component_type == "trainers":
        trainers = TRAINER_REGISTRY.list()
        if trainers:
            click.echo("\nTrainers:")
            for name in sorted(trainers):
                click.echo(f"  • {name}")
        elif component_type == "trainers":
            click.echo("  (no trainers registered)")
    
    click.echo()  # Final newline

