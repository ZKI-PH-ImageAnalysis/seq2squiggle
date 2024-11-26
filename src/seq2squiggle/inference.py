#!/usr/bin/env python

"""
Prediction of signals given a fasta file
"""
import os
import torch
import logging
import pytorch_lightning as pl
import github
import appdirs
import requests
import functools
import re
import tqdm
import shutil
from pytorch_lightning.loggers import WandbLogger

from .signal_io import BLOW5Writer, POD5Writer
from .model import seq2squiggle
from .utils import get_reads, get_profile, update_profile, update_config
from .train import DDPStrategy
from .dataloader import PoreDataModule
from . import __version__


logger = logging.getLogger("seq2squiggle")


def get_writer(
    out: str, profile: object, ideal_mode: bool, export_every_n_samples: int
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
        return BLOW5Writer(out, profile, ideal_mode), export_every_n_samples
    elif out_base.endswith(pod5_ext):
        logger.warning("POD5 Writer does not support appending to an existing file.")
        logger.warning(
            "All simulated reads will be stored in RAM before exporting to target pod5."
        )
        logger.warning(
            "This might lead to Out of Memory errors for large-scale simulations. Consider exporting to BLOW5/SLOW5 and using the blue_crab tool for conversion to pod5."
        )
        return POD5Writer(out, profile, ideal_mode), float("inf")
    else:
        logger.error("Output file must have .pod5, .slow5, or .blow5 extension.")
        raise ValueError("Output file must have .pod5, .slow5, or .blow5 extension.")


def get_saved_weights(profile_name) -> str:
    """
    Checks for the existence of the saved weights file and returns the appropriate file path.

    Parameters
    ----------
    saved_weights : str
        The path to the saved weights file.

    Returns
    -------
    str
        The path to the model weights file. Raises an exception if the file cannot be found.

    Raises
    ------
    FileNotFoundError
        If neither the specified saved weights file nor the default file in the logging directory is found.
    """
    logger.info("Weights file path is not provided.")
    cache_dir = appdirs.user_cache_dir("seq2squiggle", False, opinion=False)
    os.makedirs(cache_dir, exist_ok=True)


    # Log profile name details
    if profile_name.startswith("dna-r10"):
        logger.info("Detected R10.4.1 chemistry profile.")
        logger.info("Profile can be changed with the --profile parameter")
        profile_keyword = "R10"
    elif profile_name.startswith("dna-r9"):
        logger.info("Detected R9.4.1 chemistry profile.")
        logger.info("Profile can be changed with the --profile parameter")
        profile_keyword = "R9"
    else:
        logger.warning(
            "Profile name '%s' does not match known patterns (R10- or R9-). Proceeding with latest weights.",
            profile_name,
        )
        profile_keyword = None

    version = __version__
    version_match = None, None, 0

    # Search local cache for version- and profile-matching weights
    for filename in os.listdir(cache_dir):
        root, ext = os.path.splitext(filename)
        if ext == ".ckpt":
            file_version = tuple(
                g for g in re.match(r".*@v(\d+).(\d+).(\d+)", root).groups()
            )
            match = (
                sum(m)
                if (m := [i == j for i, j in zip(version, file_version)])[0]
                else 0
            )
            if match > version_match[2] and profile_keyword and profile_keyword in root:
                version_match = os.path.join(cache_dir, filename), None, match

    # Return best-matching local weights
    if version_match[2] > 0:
        logger.info(
            "Found matching weights in local cache: %s",
            version_match[0],
        )
        return version_match[0]

    # Search for weights on GitHub repo of seq2squiggle
    repo = github.Github().get_repo("ZKI-PH-ImageAnalysis/seq2squiggle")
    for release in repo.get_releases():
        rel_version = tuple(
            g for g in re.match(r"v(\d+)\.(\d+)\.(\d+)", release.tag_name).groups()
        )
        match = (
            sum(m) if (m := [i == j for i, j in zip(version, rel_version)])[0] else 0
        )
        if match > version_match[2]:
            for release_asset in release.get_assets():
                fn, ext = os.path.splitext(release_asset.name)
                if ext == ".ckpt":
                    if profile_keyword and profile_keyword in release_asset.name:
                        logger.info(
                            "Found matching release for %s profile: %s",
                            profile_keyword,
                            release_asset.name,
                        )
                        version_match = (
                            os.path.join(
                                cache_dir,
                                f"{fn}@v{'.'.join(map(str, rel_version))}{ext}",
                            ),
                            release_asset.browser_download_url,
                            match,
                        )
                        break
                    elif not (profile_keyword):
                        logger.info(
                            "Found no matching release for %s profile: %s",
                            profile_keyword,
                            release_asset.name,
                        )
                        # Save the latest available release for fallback
                        version_match = (
                            os.path.join(
                                cache_dir,
                                f"{fn}@v{'.'.join(map(str, rel_version))}{ext}",
                            ),
                            release_asset.browser_download_url,
                            match,
                        )
                        break
    # Download the model weights if a matching release was found.
    if version_match[2] > 0:
        filename, url, _ = version_match
        logger.info("Downloading model weights file %s from %s", filename, url)
        r = requests.get(url, stream=True, allow_redirects=True)
        r.raise_for_status()
        file_size = int(r.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.tqdm.wrapattr(
            r.raw, "read", total=file_size, desc=desc
        ) as r_raw, open(filename, "wb") as f:
            shutil.copyfileobj(r_raw, f)
        return filename
    else:
        logger.error(
            "No matching model weights for release v%s and profile %s found, please "
            "specify your model weights explicitly using the `--model` "
            "parameter",
            version,
            profile_name,
        )
        raise ValueError(
            f"No matching model weights for release v{version}  and profile {profile_name} found, "
            f"please specify your model weights explicitly using the "
            f"`--model` parameter"
        )


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

    # A list of parameter names to exclude from the comparison
    exclude_params = [
        "log_name",
        "wandb_logger_state",
        "max_chunks_train",
        "max_chunks_valid",
        "train_valid_split",
        "train_batch_size",
        "save_model",
    ]

    # Check for mismatches in parameters that are not in the exclusion list
    for param, value in architecture_params.items():
        if param not in exclude_params:
            if model_params.get(param) != value:
                if param == "seq_kmer":
                    raise ValueError(
                        f"Parameter 'seq_kmer' mismatch: Model checkpoint value is "
                        f"{model_params.get(param)}, while config value is {value}. "
                        f"The model was trained on {model_params.get(param)}-mers, while the config file expects {value}-mers. "
                        "Choose a different model or change the config value or the --profile option. "
                    )
                logger.warning(
                    f"Mismatching {param} parameter in model checkpoint "
                    f"({model_params.get(param)}) and in config file ({value})"
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
    dwell_mean: int,
    dwell_std: float,
    noise_std: float,
    noise_sampling: bool,
    duration_sampling: bool,
    distr: str,
    predict_batch_size: int,
    export_every_n_samples: int,
    sample_rate: int,
    digitisation: int,
    range_val: float,
    offset_mean: float,
    offset_std: float,
    median_before_mean: float,
    median_before_std: float,
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
    sampling_rate : int
        sampling rate
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    profile_dict = get_profile(profile)
    profile_dict = update_profile(profile_dict, sample_rate=sample_rate,
        digitisation=digitisation,
        range=range_val,
        offset_mean=offset_mean,
        offset_std=offset_std,
        median_before_mean=median_before_mean,
        median_before_std=median_before_std)

    # Update config based on profile_dict
    config = update_config(profile, config)

    ideal_mode = not(duration_sampling or dwell_std > 0)
    
    writer, export_every_n_samples = get_writer(
        out, profile_dict, ideal_mode, export_every_n_samples
    )

    if saved_weights is None:
        try:
            saved_weights = get_saved_weights(profile)
        except github.RateLimitExceededException:
            logger.error(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights. Please download compatible model weights "
                "manually from the seq2squiggle GitHub repository "
                "(https://github.com/ZKI-PH-ImageAnalysis/seq2squiggle) and specify these "
                "using the `--model` parameter"
            )
            raise PermissionError(
                "GitHub API rate limit exceeded while trying to download the "
                "model weights"
            )

    load_model = seq2squiggle.load_from_checkpoint(
        checkpoint_path=saved_weights,
        out_writer=writer,
        dwell_mean=dwell_mean,
        dwell_std=dwell_std,
        noise_std=noise_std,
        noise_sampling=noise_sampling,
        duration_sampling=duration_sampling,
        export_every_n_samples=export_every_n_samples,
    )

    check_model(load_model, config)

    reads, total_l = get_reads(fasta, read_input, n, r, c, config, distr, seed)


    # "gamma_cpu" not implemented for 'BFloat16'
    precision = "16-mixed" if torch.cuda.device_count() >= 1 else "32"

    trainer = pl.Trainer(
        accelerator="auto",
        precision=precision,
        devices="auto",
        logger=False,
        strategy=_get_strategy(),
        # use_distributed_sampler=False
    )

    rank = trainer.global_rank
    world_size = trainer.world_size

    fasta_data = PoreDataModule(
        config=config,
        data_dir=reads,
        total_l=total_l,
        batch_size=predict_batch_size,
        n_workers=1,  # n_workers > 1 causes incorrect order of IterableDataset + slower than single process
        rank=rank,
        world_size=world_size,
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


