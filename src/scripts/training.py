import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
from torch.utils.data import DataLoader, random_split
# Import custom modules
try:
    from src.scripts.utils import MODEL_TYPES, RANDOM_SEED
except ModuleNotFoundError:
    from deep_learning_project.src.scripts.utils import MODEL_TYPES, RANDOM_SEED

# Seed random number generators for reproducibility
torch.manual_seed(RANDOM_SEED)

# Define the number of workers for the dataloaders (set to 0 to use the main process)
DATALOADERS_NUM_WORKERS = 0


def train_siamese(siamese_dataset, siamese_model, max_epochs, batch_size, split_ratio, logger=None, save_path=None):

    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Split the dataset into training and evaluation sets
    train_size = int(split_ratio * len(siamese_dataset))
    validation_size = len(siamese_dataset) - train_size
    train_dataset, validation_dataset = random_split(
        siamese_dataset, [train_size, validation_size]
    )

    # Create the training and validation sets dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=DATALOADERS_NUM_WORKERS
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=DATALOADERS_NUM_WORKERS
    )

    # Set the model to training mode (if not already)
    siamese_model.train()

    # Get the model's checkpoint folder and name
    checkpoint_folder = "/".join(save_path.split("/")[:-1]) + "/"
    checkpoint_name = save_path.split("/")[-1]
    print("checkpoint_folder:", checkpoint_folder)
    print("checkpoint_name:", checkpoint_name)

    # Train the model (using the PyTorch Lightning's Trainer)
    trainer = pl.Trainer(
        # Set the maximum number of epochs
        max_epochs=max_epochs,
        # Avoids executing a validation sanity check at the beginning of the training (to speed up the training process)
        #   NOTE: this has no effect on the training, as this check is only used to verify if errors on the validation set
        #   are present before starting training (rather than during training itself)
        num_sanity_val_steps=0,
        # Set the logger if provided
        logger=logger,
    )
    trainer.fit(siamese_model, train_dataloader, validation_dataloader)

    # Get the wandb run ID
    wandb_run_id = None
    if logger is not None:
        wandb_run_id = logger.experiment.id
        # Finish the current run
        logger.experiment.finish(quiet=True)

    # Save model checkpoints
    trainer.save_checkpoint(save_path)

    # Return datasets
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "run_id": wandb_run_id
    }


def train_transformer(transformer_indexing_dataset, transformer_retrieval_dataset, transformer_model: pl.LightningModule, max_epochs_list: list[int, int], batch_size, indexing_split_ratios: tuple[float, float], retrieval_split_ratios: tuple[float, float, float], logger: WandbLogger, save_path: str):
    '''
    Train the transformer model for the indexing and retrieval tasks.

    Args:
    - transformer_indexing_dataset (Dataset): The dataset for the indexing task.
    - transformer_retrieval_dataset (Dataset): The dataset for the retrieval task.
    - transformer_model (pl.LightningModule): The transformer model to train.
    - max_epochs_list (list[int, int]): List containing the maximum number of epochs for the indexing and retrieval tasks, respectively.
    - batch_size (int): The batch size for the dataloaders.
    - indexing_split_ratios (tuple[float, float]): The split ratios for the indexing dataset (training and validation).
    - retrieval_split_ratios (tuple[float, float, float]): The split ratios for the retrieval dataset (training, validation, and test).
    - logger (pl.loggers.WandbLogger): The logger for the training process.
    - save_path (str): The path to save the model checkpoints.
    '''

    # Split the retrieval dataset into training and evaluation sets for the retrieval task
    indexing_train_size = \
        int(indexing_split_ratios[0] * len(transformer_indexing_dataset))
    indexing_validation_size = \
        len(transformer_indexing_dataset) - indexing_train_size
    indexing_train_dataset, indexing_validation_dataset = random_split(
        transformer_indexing_dataset, [
            indexing_train_size, indexing_validation_size]
    )

    # Create the training and validation sets dataloaders for the retrieval task
    indexing_train_dataloader = DataLoader(
        indexing_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=DATALOADERS_NUM_WORKERS
    )
    indexing_validation_dataloader = DataLoader(
        indexing_validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=DATALOADERS_NUM_WORKERS
    )

    # Split the retrieval dataset into training and evaluation sets for the retrieval task
    retrieval_train_size = \
        int(retrieval_split_ratios[0] * len(transformer_retrieval_dataset))
    retrieval_validation_size = \
        int(retrieval_split_ratios[1] * len(transformer_retrieval_dataset))
    retrieval_test_size = len(transformer_retrieval_dataset) - \
        retrieval_train_size - retrieval_validation_size
    retrieval_train_dataset, retrieval_validation_dataset, retrieval_test_dataset = random_split(
        transformer_retrieval_dataset, [
            retrieval_train_size,
            retrieval_validation_size,
            retrieval_test_size
        ]
    )

    # Create the training and validation sets dataloaders for the retrieval task
    retrieval_train_dataloader = DataLoader(
        retrieval_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=DATALOADERS_NUM_WORKERS
    )
    retrieval_validation_dataloader = DataLoader(
        retrieval_validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=DATALOADERS_NUM_WORKERS
    )
    retrieval_test_dataloader = DataLoader(
        retrieval_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=DATALOADERS_NUM_WORKERS
    )

    # Set the model to training mode (if not already)
    transformer_model.train()

    # Get the model's checkpoint folder and name
    checkpoint_folder = "/".join(save_path.split("/")[:-1]) + "/"
    checkpoint_name = save_path.split("/")[-1]
    print("checkpoint_folder:", checkpoint_folder)
    print("checkpoint_name:", checkpoint_name)

    # Train the model (using the PyTorch Lightning's Trainer) for the indexing task
    print("Training the model for the indexing task...")
    trainer = pl.Trainer(
        # Set the maximum number of epochs
        max_epochs=max_epochs_list[0],
        # Avoids executing a validation sanity check at the beginning of the training (to speed up the training process)
        #   NOTE: this has no effect on the training, as this check is only used to verify if errors on the validation set
        #   are present before starting training (rather than during training itself)
        num_sanity_val_steps=0,
        # Set the logger if provided
        logger=logger[0],
    )
    trainer.fit(transformer_model, indexing_train_dataloader,
                indexing_validation_dataloader)
    print("Trained the model for the indexing task.")
    indexing_run_id = None
    if logger is not None:
        # Get the wandb run ID
        indexing_run_id = logger[0].experiment.id
        # Finish the current run
        logger[0].experiment.finish(quiet=True)

    # Reset the model's schdeduled sampling probability for the retrieval task
    transformer_model.reset_scheduled_sampling_probability()

    # Train the model (using the PyTorch Lightning's Trainer) for the retrieval task
    print("Training the model for the retrieval task...")
    trainer = pl.Trainer(
        # Set the maximum number of epochs
        max_epochs=max_epochs_list[1],
        # Avoids executing a validation sanity check at the beginning of the training (to speed up the training process)
        #   NOTE: this has no effect on the training, as this check is only used to verify if errors on the validation set
        #   are present before starting training (rather than during training itself)
        num_sanity_val_steps=0,
        # Set the logger if provided
        logger=logger[1]
    )
    trainer.fit(transformer_model, retrieval_train_dataloader,
                retrieval_validation_dataloader)
    print("Trained the model for the retrieval task.")
    retrieval_run_id = None
    if logger is not None:
        # Get the wandb run ID
        retrieval_run_id = logger[1].experiment.id
        # Finish the current run
        logger[1].experiment.finish(quiet=True)

    # Save model checkpoints
    trainer.save_checkpoint(save_path)

    # Return datasets
    return {
        "indexing": {
            "train": indexing_train_dataset,
            "validation": indexing_validation_dataset
        },
        "retrieval": {
            "train": retrieval_train_dataset,
            "validation": retrieval_validation_dataset,
            "test": retrieval_test_dataset
        },
        "run_ids": {
            "indexing": indexing_run_id,
            "retrieval": retrieval_run_id
        }
    }
