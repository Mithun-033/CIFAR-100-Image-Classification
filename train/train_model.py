import torch.nn as nn
import torch.optim as optim
from rich import print
from rich.panel import Panel

from model.model import Model
from model.config import training_config
from data.dataloaders import get_dataloaders


def train_model(model: Model, config: training_config):
    """
    Train the model using the provided configuration.

    Args:
        model (Model): The model to be trained.
        config (training_config): The training configuration parameters.

    Returns:
        None
    """
    # Get the dataloaders for training and validation
    train_loader, val_loader = get_dataloaders(config.batch_size, config.num_workers)

    # Define the optimizer and loss function
    optimizer = ...
    criterion = ...

    # Training loop
    print(Panel.Panel("Starting Training", style="bold green"))
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            # Implement the training step here
            ...

        model.eval()
        for batch in val_loader:
            # Validation step can be added here
            ...