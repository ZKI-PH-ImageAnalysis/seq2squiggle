#!/usr/bin/env python

import logging
import pathlib
import yaml
import os
import rich_click as click
import wandb

from train import train_run
from preprocess import preprocess_run
from inference import inference_run
from train_sweep import train_sweep_run
from utils import set_seeds, setup_logging

logger = logging.getLogger("seq2squiggle")

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.WIDTH = None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class _SharedParams(click.RichCommand):
    """Options shared between most seq2squiggle commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("-s", "--seed"),
                help="""
                Set the seed value for reproducibility
                """,
                type=int,
                default=385,
            ),
            click.Option(
                ("-m", "--model"),
                help="""
                The model weights (.ckpt file). If not provided, seq2squiggle
                will try to download the latest release.
                """,
                type=click.Path(exists=False, dir_okay=False),
            ),
            click.Option(
                ("-y", "--config"),
                help="""
                The YAML configuration file overriding the default options.
                Default is config/config.yaml
                """,
                default="config/config.yaml",
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-v", "--verbosity"),
                help="""
                Set the verbosity of console logging messages. Log files are
                always set to 'info'.
                """,
                type=click.Choice(
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """
    # seq2squiggle

    seq2squiggle predicts nanopore sequencing signals using a Feed-Forward Transformer.
    seq2squiggle supports fasta/q files for signal prediction and events.tsv from uncalled4 for training new models.
    For more information check the official code repository:
    - [https://github.com/ZKI-PH3/seq2squiggle]()

    Please cite the following publication if you use seq2squiggle in your work:
    - Beslic, D., Kucklick, M., Engelmann, S., Fuchs, S., Renards, B.Y., Körber, N. End-to-end simulation of nanopore sequencing signals with feed-forward transformers. bioRxiv (2024).
    """


@main.command(cls=_SharedParams)
@click.argument(
    "events_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "outdir",
    required=True,
    type=click.Path(dir_okay=True, file_okay=False),
)
@click.option(
    "--no_batches",
    is_flag=True,
    default=True,
    show_default=True,
    help="Process the events.tsv file without reading it in batches",
)
@click.option(
    "--chunksize",
    type=int,
    show_default=True,
    default=100000,
    help="Specify the chunk size for each batch.",
)
def preprocess(
    events_path,
    outdir,
    no_batches,
    chunksize,
    seed,
    model,
    config,
    verbosity,
):
    """
    Preprocess f5c's events.tsv for training the model

    EVENTS_PATH must be a events.tsv from f5c.
    OUTDIR must be path to output directory
    """
    setup_logging(verbosity)
    set_seeds(seed)
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    preprocess_run(
        events_path=events_path,
        outdir=outdir,
        batches=no_batches,
        chunksize=chunksize,
        config=config,
    )
    logger.info("Preprocessing done.")


@main.command(cls=_SharedParams)
@click.argument(
    "train_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
)
@click.argument(
    "valid_dir",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    required=False,
)
@click.option(
    "--save_valid_plots",
    default=True,
    type=bool,
    help="Save validation plots during training if set to True.",
)
def train(
    train_dir,
    valid_dir,
    save_valid_plots,
    seed,
    model,
    config,
    verbosity,
):
    """
    Train the model with pre-processed npz chunks

    NPY_DIR must be directory containing the .npy files from the preprocessing module
    """
    setup_logging(verbosity)
    set_seeds(seed)
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    # print all parameters defined in config file
    logger.info("Config parameters:")
    for key in config:
        logger.info(f" {key}: {config[key]}")

    train_run(
        train_dir=train_dir,
        valid_dir=valid_dir,
        config=config,
        model_path=model,
        save_valid_plots=save_valid_plots,
    )
    logger.info("Training done.")


@main.command(cls=_SharedParams)
@click.argument(
    "fasta",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
)
@click.option(
    "--read-input",
    default=False,
    is_flag=True,
    show_default=True,
    help="Disable read generation if the input FASTA/Q file contains reads instead of a single genome.",
)
@click.option(
    "-n",
    "--num-reads",
    type=int,
    default=-1,
    help="Specify the desired number of generated reads.",
)
@click.option(
    "-r",
    "--read-length",
    type=int,
    default=10000,
    show_default=True,
    help="Specify the desired average read length.",
)
@click.option(
    "-c",
    "--coverage",
    type=int,
    default=-1,
    help="Specify the desired genome coverage.",
)
@click.option(
    "-o",
    "--out",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Specify the path to the output POD5/SLOW5/BLOW5 file.",
)
@click.option(
    "--profile",
    default="prom_r10_dna",
    show_default=True,
    type=click.Choice(["minion_r10_dna", "prom_r10_dna"]),
    help="Select a profile for data simulation. The profile determines values for digitization, sample rate, range, offset mean, offset standard deviation, median before mean, and median before standard deviation.",
)
@click.option(
    "--noise-sampler",
    show_default=True,
    default=True,
    type=bool,
    help="Enable or disable the noise sampler. If disabled, no noise will be added to the signal.",
)
@click.option(
    "--duration-sampler",
    show_default=True,
    default=True,
    type=bool,
    help="Enable or disable the duration sampler. If disabled, the ideal event length will be used.",
)
@click.option(
    "--ideal-event-length",
    default=-1.0,
    show_default=True,
    type=float,
    help="Specify the ideal event length to use. This option is only effective if the duration sampler is disabled. If set to -1, a static normal distribution will be used.",
)
@click.option(
    "--noise-std",
    default=1.0,
    show_default=True,
    type=float,
    help="Set the standard deviation for noise. When the noise sampler is enabled, the noise generated will be scaled by this value. If the noise sampler is disabled, a static normal distribution will be used. No additional noise will be added if noise-std is less than or equal to 0.",
)
@click.option(
    "--distr",
    default="expon",
    show_default=True,
    type=click.Choice(["expon", "beta", "gamma"]),
    help="Choose a distribution for read sampling. This option is only required in genome mode.",
)
@click.option(
    "--predict-batch-size",
    default=1024,
    show_default=True,
    type=int,
    help="Specify the batch size for prediction.",
)
@click.option(
    "--export-every-n-samples",
    default=500000,
    show_default=True,
    type=int,
    help="Specify how often the predicted samples (chunk) should be saved to output file. Increasing it will reduce runtime and increase memory consumption.",
)
def predict(
    fasta,
    read_input,
    num_reads,
    read_length,
    coverage,
    out,
    profile,
    noise_sampler,
    duration_sampler,
    ideal_event_length,
    noise_std,
    distr,
    predict_batch_size,
    export_every_n_samples,
    seed,
    model,
    config,
    verbosity,
):
    """
    Generate sequencing signals from genome or read fasta file

    FASTA must be .fasta file with desired genome or reads for simulation
    """

    setup_logging(verbosity)

    # Collect arguments into a dictionary
    args = {
        "fasta": fasta,
        "read_input": read_input,
        "num_reads": num_reads,
        "read_length": read_length,
        "coverage": coverage,
        "out": out,
        "profile": profile,
        "noise_sampler": noise_sampler,
        "duration_sampler": duration_sampler,
        "ideal_event_length": ideal_event_length,
        "noise_std": noise_std,
        "distr": distr,
        "predict_batch_size": predict_batch_size,
        "export_every_n_samples": export_every_n_samples,
        "seed": seed,
        "model": model,
        "config": config,
        "verbosity": verbosity,
    }
    # Log all arguments
    logger.info("Arguments:")
    for key, value in args.items():
        logger.info(f" {key}: {value}")
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    logger.debug("Config parameters:")
    for key in config:
        logger.debug(f" {key}: {config[key]}")

    set_seeds(seed)

    inference_run(
        config=config,
        saved_weights=model,
        fasta=fasta,
        read_input=read_input,
        n=num_reads,
        r=read_length,
        c=coverage,
        out=out,
        profile=profile,
        ideal_event_length=ideal_event_length,
        noise_std=noise_std,
        noise_sampling=noise_sampler,
        duration_sampling=duration_sampler,
        distr=distr,
        predict_batch_size=predict_batch_size,
        export_every_n_samples=export_every_n_samples,
        seed=seed,
    )
    logger.info("Prediction done.")


@click.option(
    "--sweep-id",
    type=str,
    required=True,
    help="Sweep id. Should look like 'user/project/id'",
)
@main.command(cls=_SharedParams)
def sweep(
    sweep_id,
    seed,
    model,
    config,
    verbosity,
):
    """
    Perform a sweep training

    To create a new sweep you need to create a wandb sweep
    'wandb sweep --project projectname configs/sweep.yaml'
    """
    setup_logging(verbosity)
    set_seeds(seed)

    with open(config) as f_in:
        sweep_config = yaml.safe_load(f_in)
    wandb.agent(sweep_id, train_sweep_run, count=200)


def download_model_weights():
    """
    Use cached model weights or download them from GitHub.

    Returns
    -------
    str
        The name of the model weights file.
    """
    logger.error("Not implemented yet.")
    exit()


if __name__ == "__main__":
    main()
