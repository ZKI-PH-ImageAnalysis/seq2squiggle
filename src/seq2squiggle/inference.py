#!/usr/bin/env python

"""
Prediction of signals given a fasta file
"""
import os
import torch
import logging
from signal_io import BLOW5Writer, POD5Writer
from model import seq2squiggle
import pytorch_lightning as pl

from utils import get_reads
from train import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from dataloader import PoreDataModule


logger = logging.getLogger("seq2squiggle")


def get_writer(
    out: str, profile: object, ideal_event_length: int, export_every_n_samples: int
) -> tuple:
    """
    Returns an appropriate file writer object based on the output file extension.

    Parameters
    ----------
    out : str
        The path to the output file.
    profile : object
        The profile object used for writing data.
    ideal_event_length : int
        The ideal length of the event.
    export_every_n_samples : int
        The number of samples after which to export data.

    Returns
    -------
    tuple
        A tuple containing the file writer object and the number of samples before export.
    """
    slow5_ext = [".blow5", ".slow5"]
    pod5_ext = ".pod5"
    out_base = os.path.basename(out)

    if os.path.exists(out):
        logger.warning(f"Output file {out} already exists. File will be deleted.")
        os.remove(out)

    if any(out_base.endswith(ext) for ext in slow5_ext):
        return BLOW5Writer(out, profile, ideal_event_length), export_every_n_samples
    elif out_base.endswith(pod5_ext):
        logger.warning("POD5 Writer does not support appending to an existing file.")
        logger.warning(
            "All simulated reads will be stored in RAM before exporting to target pod5."
        )
        logger.warning(
            "This might lead to Out of Memory errors for large-scale simulations. Consider exporting to BLOW5/SLOW5 and using the blue_crab tool for conversion to pod5."
        )
        return POD5Writer(out, profile, ideal_event_length), float("inf")
    else:
        logger.error("Output file must have .pod5, .slow5, or .blow5 extension.")
        raise ValueError("Output file must have .pod5, .slow5, or .blow5 extension.")


def check_savedweights(saved_weights: str, log_name: str) -> str:
    """
    Checks for the existence of the saved weights file and returns the appropriate file path.

    Parameters
    ----------
    saved_weights : str
        The path to the saved weights file.
    log_name : str
        The name used to construct the logging directory path.

    Returns
    -------
    str
        The path to the model weights file. Raises an exception if the file cannot be found.

    Raises
    ------
    FileNotFoundError
        If neither the specified saved weights file nor the default file in the logging directory is found.
    """
    log_dir = "./logs-" + log_name
    if saved_weights and os.path.isfile(saved_weights):
        model_path = saved_weights
    elif os.path.isfile(os.path.join(log_dir, "last.ckpt")):
        logger.warning("Model weights were not found or were not set via --model.")
        logger.warning(
            f"Model weights from logging directory {log_dir} will be used instead."
        )
        model_path = os.path.join(log_dir, "last.ckpt")
    else:
        logger.error("Output filemust have .slow5 or .blow5 extension.")
        raise FileNotFoundError(
            "Model weights could not be found. "
            "Please use the --model argument to specify the path to your model file."
        )

    return model_path


def check_model(model: object, config: dict) -> None:
    """
    Verifies that the model parameters match the configuration settings.

    Parameters
    ----------
    model : object
        The model object containing hyperparameters and architecture settings.
    config : dict
        A dictionary containing the expected configuration parameters.

    Returns
    -------
    None
    """
    model_params = model.hparams.config
    architecture_params = config

    for param in architecture_params:
        if model_params[param] != architecture_params[param]:
            logger.warning(
                f"Mismatching {param} parameter in model checkpoint"
                f" ({model_params[param]}) and in config file ({architecture_params[param]})"
            )


def inference_run(
    config: dict,
    saved_weights: str,
    fasta: str,
    read_input: bool,
    n: int,
    r: int,
    c: int,
    out: str,
    profile: dict,
    ideal_event_length: int,
    noise_std: float,
    noise_sampling: bool,
    duration_sampling: bool,
    distr: str,
    predict_batch_size: int,
    export_every_n_samples: int,
    seed: int,
):
    """
    Runs the inference process for nanopore sequencing signal prediction.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration parameters for the inference.
    saved_weights : str
        Path to the saved model weights file.
    fasta : str
        Path to the FASTA file containing sequence data.
    read_input : bool
        Flag indicating whether to read input data.
    n : int
        Number of samples.
    r : int
        Number of rows (or another dimension parameter).
    c : int
        Number of columns (or another dimension parameter).
    out : str
        Path to the output file where predictions will be saved.
    profile : dict
        Profile object used for writing data.
    ideal_event_length : int
        The ideal length of the event.
    noise_std : float
        Standard deviation of noise to be added.
    noise_sampling : bool
        Flag indicating whether noise sampling is enabled.
    duration_sampling : bool
        Flag indicating whether duration sampling is enabled.
    distr : str
        Distribution type for the sampling.
    predict_batch_size : int
        Batch size for predictions.
    export_every_n_samples : int
        Number of samples after which to export data.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    writer, export_every_n_samples = get_writer(
        out, profile, ideal_event_length, export_every_n_samples
    )

    saved_weights = check_savedweights(saved_weights, config["log_name"])

    load_model = seq2squiggle.load_from_checkpoint(
        checkpoint_path=saved_weights,
        out_writer=writer,
        ideal_event_length=ideal_event_length,
        noise_std=noise_std,
        noise_sampling=noise_sampling,
        duration_sampling=duration_sampling,
        export_every_n_samples=export_every_n_samples,
    )
    check_model(load_model, config)

    reads, total_l = get_reads(fasta, read_input, n, r, c, config, distr, seed)

    fasta_data = PoreDataModule(
        config=config,
        data_dir=reads,
        total_l=total_l,
        batch_size=predict_batch_size,
        n_workers=1,  # n_workers > 1 causes incorrect order of IterableDataset + slower than single process
    )

    wandb_logger = WandbLogger(
        project="seq2squiggle-testing",
        config=config,
        name=config["log_name"],
        mode="disabled",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        precision="16-mixed",
        devices="auto",
        logger=wandb_logger,
        strategy=_get_strategy(),
    )

    trainer.predict(model=load_model, datamodule=fasta_data, return_predictions=False)


def _get_strategy():
    """
    Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)
    return "auto"
