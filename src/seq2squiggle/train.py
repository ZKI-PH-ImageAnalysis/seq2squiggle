#!/usr/bin/env python

"""
Training Loop
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from typing import Union
import os
import logging

from .dataloader import PoreDataModule
from .utils import count_parameters, n_workers
from .model import seq2squiggle

logger = logging.getLogger("seq2squiggle")

TORCH_CUDNN_V8_API_ENABLED = 1
torch.set_float32_matmul_precision("medium")


def train_run(
    train_dir: str,
    valid_dir: str,
    config: dict,
    model_path: str,
    save_valid_plots: bool,
) -> None:
    fft_model = seq2squiggle(config=config, save_valid_plots=save_valid_plots)

    count_parameters(fft_model)

    poredata = PoreDataModule(
        config=config,
        data_dir=train_dir,
        valid_dir=valid_dir,
        batch_size=config["train_batch_size"],
        n_workers=max(n_workers(), 4),
    )

    wandb_logger = WandbLogger(
        project="seq2squiggle-testing",
        config=config,
        name=config["log_name"],
        mode=config["wandb_logger_state"],
    )

    # Use model_dir as Log directory
    if model_path is not None:
        log_dir, filename = os.path.split(model_path)

        if log_dir and ".ckpt" in filename:
            ckpt_n, _ = os.path.splitext(filename)
        else:
            logger.error(
                "Please specify the model path as follows MODEL_DIR/MODEL.CKPT"
            )
            raise ValueError(
                "Invalid model path. The model path should be specified as follows: MODEL_DIR/MODEL.CKPT"
            )
    else:
        log_dir = "./logs-" + config["log_name"]
        ckpt_n = "last"
        logger.info(
            f"Model path not specified. Model will be saved to {log_dir} as {ckpt_n}.ckpt"
        )
        model_path = os.path.join(log_dir, ckpt_n + ".ckpt")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.exists(model_path):
        logger.warning(
            f"{model_path} already exists. The name of the saved model will be appended with a version count starting with v1 to avoid collisions."
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=log_dir,
            save_top_k=-1,
            save_weights_only=True,
            every_n_epochs=10,
            filename=ckpt_n,
        ),
        LearningRateMonitor(
            logging_interval="step", log_momentum=True, log_weight_decay=True
        ),
    ]

    # "gamma_cpu" not implemented for 'BFloat16'
    precision = "16-mixed" if torch.cuda.device_count() >= 1 else "64"

    trainer = pl.Trainer(
        accelerator="auto",
        precision=precision,
        devices="auto",
        callbacks=callbacks,
        default_root_dir=log_dir,
        enable_checkpointing=True,
        max_epochs=config["max_epochs"],
        logger=wandb_logger,
        gradient_clip_val=config["gradient_clip_val"],
        strategy=_get_strategy(),
    )

    trainer.fit(fft_model, poredata)

    logger.info("Training finished.")


def _get_strategy() -> Union[str, DDPStrategy]:
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
