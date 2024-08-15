#!/usr/bin/env python

"""
Training Loop
"""
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from .dataloader import PoreDataModule
from .model import seq2squiggle
from .train import _get_strategy


def train_sweep_run(npz_path="preprocess-HP-run/"):
    with wandb.init(config=None):
        config = wandb.config

        model = seq2squiggle(config=config, most_common_numbers=None)

        poredata = PoreDataModule(
            config, data_dir=npz_path, batch_size=config["train_batch_size"]
        )

        wandb_logger = WandbLogger(
            project="sweep_loss",
            config=config,
            name=config["log_name"],
            mode=config["wandb_logger_state"],
        )
        log_dir = "./logs-" + config["log_name"]

        trainer = pl.Trainer(
            accelerator="auto",
            precision="16-mixed",
            default_root_dir=log_dir,
            enable_checkpointing=False,
            max_epochs=config["max_epochs"],
            logger=wandb_logger,
            gradient_clip_val=config["gradient_clip_val"],
            detect_anomaly=False,
            deterministic=True,
            strategy=_get_strategy(),
        )

        trainer.fit(model, poredata)
