#!/usr/bin/env python

import logging
import pathlib
import yaml
import os
import rich_click as click
import wandb
import warnings
import torch
import pytorch_lightning as pl
import pod5

from .train import train_run
from .preprocess import preprocess_run
from .inference import inference_run
from .train_sweep import train_sweep_run
from .utils import set_seeds, setup_logging
from . import __version__


warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)
warnings.filterwarnings(
    "ignore",
    ".*In combination with multi-process data loading*",
)
warnings.filterwarnings(
    "ignore",
    ".*predict returned None if it was on purpose*",
)

logger = logging.getLogger("seq2squiggle")

click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.WIDTH = None


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
                """,
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
    - [https://github.com/ZKI-PH-ImageAnalysis/seq2squiggle]()

    Please cite the following publication if you use seq2squiggle in your work:
    - Beslic D, Kucklick M, Engelmann S, Fuchs S, Renard BY, Körber N. End-to-end simulation of nanopore sequencing signals with feed-forward transformers. *Bioinformatics*. 2024; btae744. [https://doi.org/10.1093/bioinformatics/btae744]() 
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
    Preprocess uncalled4's events.tsv for training the model

    EVENTS_PATH must be a events.tsv from uncalled4 or f5c.
    OUTDIR must be path to output directory
    """
    setup_logging(verbosity)
    logger.info("seq2squiggle version %s", str(__version__))
    set_seeds(seed)
    config = set_config(config)
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
    logger.info("seq2squiggle version %s", str(__version__))
    set_seeds(seed)
    config = set_config(config)
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


# Function to conditionally show advanced options
def conditional_option(f):
    f = click.option(
        "--noise-sampler",
        default=True,
        type=bool,
        help="Enable or disable the noise sampler.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--duration-sampler",
        default=True,
        type=bool,
        help="Enable or disable the duration sampler.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--dwell-mean",
        default=9.0,
        type=float,
        help="Specify the mean dwell time (=number of signal points per k-mer). This will only be used if the duration sampler is deactivated",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--dwell-std",
        default=0.0,
        type=float,
        help="Specify the standard deviation of the dwell time (=number of signal points per k-mer). This will only be used if the duration sampler is deactivated",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--noise-std",
        default=1.0,
        type=float,
        help="Set the standard deviation for noise.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--distr",
        default="expon",
        type=click.Choice(["expon", "beta", "gamma"]),
        help="Choose a distribution for read sampling.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--predict-batch-size",
        default=1024,
        type=int,
        help="Specify the batch size for prediction.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--export-every-n-samples",
        default=1000000,
        type=int,
        help="Specify how often the predicted samples should be saved.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--sample-rate",
        default=5000,
        type=int,
        help="Specify the sampling rate.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--digitisation",
        default=None,
        type=int,
        help="Specify the digitisation.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--range_val",
        default=None,
        type=float,
        help="Specify the range value.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--offset_mean",
        default=None,
        type=float,
        help="Specify the digitisation.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--offset_std",
        default=None,
        type=float,
        help="Specify the digitisation.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--median_before_mean",
        default=None,
        type=float,
        help="Specify the digitisation.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    f = click.option(
        "--median_before_std",
        default=None,
        type=float,
        help="Specify the digitisation.",
        show_default=True,
        hidden=True  # Hidden by default
    )(f)
    return f





@main.command(cls=_SharedParams, context_settings={"ignore_unknown_options": True})
@click.argument(
    "fasta",
    required=False,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
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
    required=False,
    type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Specify the path to the output POD5/SLOW5/BLOW5 file.",
)
@click.option(
    "--profile",
    default="dna-r10-prom",
    show_default=True,
    type=click.Choice(["dna-r10-prom", "dna-r10-min", "dna-r9-prom", "dna-r9-min"]),
    help="Select a profile for data simulation. The profile determines values for digitization, sample rate, range, offset mean, offset standard deviation, median before mean, and median before standard deviation.",
)
@click.option(
    "--show-advanced-options", 
    is_flag=True, 
    default=False, 
    help="Show advanced options for signal prediction."
)
@conditional_option
@click.pass_context
def predict(
    ctx,
    fasta,
    read_input,
    num_reads,
    read_length,
    coverage,
    out,
    profile,
    show_advanced_options,
    noise_sampler,
    duration_sampler,
    dwell_mean,
    dwell_std,
    noise_std,
    distr,
    predict_batch_size,
    export_every_n_samples,
    sample_rate,
    digitisation,
    range_val,
    offset_mean,
    offset_std,
    median_before_mean,
    median_before_std,
    seed,
    model,
    config,
    verbosity,
):
    """
    Generate sequencing signals from genome or read fasta file

    FASTA must be .fasta file with desired genome or reads for simulation
    """
    if show_advanced_options:
        # Dynamically re-generate the command's help message with hidden=False
        for param in ctx.command.params:
            param.hidden = False

        # Re-run help message to show advanced options
        click.echo(ctx.get_help())
        ctx.exit()  # Exit after showing help with advanced options
    
    # Check for help flag
    if ctx.invoked_subcommand is None and ctx.args and "-h" in ctx.args:
        # Print the normal help and exit
        click.echo(ctx.get_help())
        ctx.exit()

    if not fasta or not out:
        logger.error("FASTA file and Output file are required for prediction.")
        ctx.exit(1)


    setup_logging(verbosity)
    logger.info("seq2squiggle version %s", str(__version__))

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
        "dwell_mean": dwell_mean,
        "dwell_std": dwell_std,
        "noise_std": noise_std,
        "distr": distr,
        "predict_batch_size": predict_batch_size,
        "export_every_n_samples": export_every_n_samples,
        "sample_rate": sample_rate,
        "digitisation": digitisation,
        "range": range_val,
        "offset_mean": offset_mean,
        "offset_std": offset_std,
        "median_before_mean": median_before_mean,
        "median_before_std": median_before_std,
        "seed": seed,
        "model": model,
        "config": config,
        "verbosity": verbosity,
    }
    # Log all arguments
    logger.info("Arguments:")
    for key, value in args.items():
        logger.info(f" {key}: {value}")

    config = set_config(config)
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
        dwell_mean=dwell_mean,
        dwell_std=dwell_std,
        noise_std=noise_std,
        noise_sampling=noise_sampler,
        duration_sampling=duration_sampler,
        distr=distr,
        predict_batch_size=predict_batch_size,
        export_every_n_samples=export_every_n_samples,
        sample_rate=sample_rate,
        digitisation=digitisation,
        range_val=range_val,
        offset_mean=offset_mean,
        offset_std=offset_std,
        median_before_mean=median_before_mean,
        median_before_std=median_before_std,
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
    logger.info("seq2squiggle version %s", str(__version__))
    set_seeds(seed)
    config = set_config(config)
    wandb.agent(sweep_id, train_sweep_run, count=200)


@main.command()
def version():
    """Get the version of seq2squiggle"""
    setup_logging("info")
    logger.info(f"seq2squiggle: {__version__}")
    logger.info(f"pytorch: {torch.__version__}")
    logger.info(f"lightning: {pl.__version__}")
    logger.info(f"pod5: {pod5.__version__}")



def set_config(config_path : dict) -> dict:
    default_config_path = pathlib.Path(__file__).parent / "config.yaml"
    path_to_use = default_config_path if config_path is None else config_path

    try:
        with open(path_to_use, 'r') as f_in:
            config = yaml.safe_load(f_in)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {path_to_use}")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {path_to_use} - {exc}")
        raise

    if config_path is None:
        logger.info(f"Config file was not specified. Default config will be used.")

    return config
    
if __name__ == "__main__":
    main()
