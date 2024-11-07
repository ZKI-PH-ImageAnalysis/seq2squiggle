#!/usr/bin/env python

"""
Model architecture
"""

import torch
from collections import defaultdict
from torch import nn
import transformers
import pytorch_lightning as pl
from transformers.optimization import AdafactorSchedule
import logging
import numpy as np

from .modules import Encoder, LengthRegulator, Decoder, NoiseSampler
from .utils import generate_validation_plots
from .signal_io import BLOW5Writer

logger = logging.getLogger("seq2squiggle")

torch.set_float32_matmul_precision("medium")


class seq2squiggle(pl.LightningModule):
    """
    Transformer Feed Forward model for predicting nanopore signals.
    """

    def __init__(
        self,
        *,
        config: dict,
        save_valid_plots: bool = True,
        out_writer: None = None,
        dwell_mean: float = 9.0,
        dwell_std: float = 0.0,
        noise_std: float = -1,
        noise_sampling: bool = False,
        duration_sampling: bool = False,
        export_every_n_samples: int = 2000000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoders = Encoder(config)
        self.length_regulator = LengthRegulator(config)
        self.decoders = Decoder(config)
        self.noise_sampler = NoiseSampler(config)
        self.config = config
        self.save_valid_plots = save_valid_plots
        self.results = []
        self.out_writer = out_writer
        self.dwell_mean = dwell_mean
        self.dwell_std = dwell_std
        self.noise_std = noise_std
        self.noise_sampling = noise_sampling
        self.duration_sampling = duration_sampling
        self.export_every_n_samples = export_every_n_samples
        self.total_samples = 0

    def training_step(self, batch, batch_idx):
        data, targets, data_ls, targets_ls, noise_std, *args = batch

        (
            bs,
            seq_l,
        ) = (
            data.shape[0],
            data.shape[1],
        )
        data = data.reshape(bs, seq_l, -1)

        enc_out, emb_out = self.encoders(data)

        noise_std_prediction = self.noise_sampler(emb_out.detach().clone())
        noise_std_prediction = noise_std_prediction[:, :, None]

        length_predict_out, duration_pred_out, dist_duration, _ = self.length_regulator(
            emb_out=emb_out.detach().clone(),
            x=enc_out,
            target=data_ls,
            noise_std_prediction=noise_std_prediction,
            max_length=self.config["max_signal_len"],
        )

        prediction = self.decoders(length_predict_out)

        loss_dict = get_loss(
            self=self,
            prediction=prediction,
            targets=targets,
            duration_pred_out=duration_pred_out,
            data_ls=data_ls,
            dist_duration=dist_duration,
            noise_std_prediction=noise_std_prediction,
            noise_std=noise_std,
            step="train",
        )

        return loss_dict

    def validation_step(self, batch, batch_idx):
        data, targets, data_ls, targets_ls, noise_std, *args = batch

        (
            bs,
            seq_l,
        ) = (
            data.shape[0],
            data.shape[1],
        )
        data = data.reshape(bs, seq_l, -1)

        enc_out, emb_out = self.encoders(data)

        noise_std_prediction = self.noise_sampler(emb_out.detach().clone())
        noise_std_prediction = noise_std_prediction[:, :, None]

        # signal with ground-truth duration for comparable validation loss
        (
            length_predict_out,
            duration_pred_ref,
            dist_duration,
            noise_std_prediction_ext,
        ) = self.length_regulator(
            emb_out=emb_out,
            x=enc_out,
            target=data_ls,
            noise_std_prediction=noise_std_prediction,
            max_length=self.config["max_signal_len"],
        )

        prediction = self.decoders(length_predict_out)

        get_loss(
            self=self,
            prediction=prediction,
            targets=targets,
            duration_pred_out=duration_pred_ref,
            data_ls=data_ls,
            dist_duration=dist_duration,
            noise_std_prediction=noise_std_prediction,
            noise_std=noise_std,
            step="valid",
        )

        prediction_idealtime = prediction.detach().clone()

        # signal with predicted duration as output
        length_predict_out, duration_pred_out, dist_duration, _ = self.length_regulator(
            emb_out,
            enc_out,
            target=None,
            max_length=self.config["max_signal_len"],
            noise_std_prediction=noise_std_prediction,
        )
        prediction = self.decoders(length_predict_out)

        if batch_idx == 0:
            # Rescale target to raw values using a fixed scaling value
            targets = targets.detach().cpu().squeeze()
            targets = targets * self.config["scaling_max_value"]
            # Rescale preds to raw values using a fixed scaling value
            prediction = prediction.detach().cpu().squeeze()
            prediction = prediction * self.config["scaling_max_value"]
            prediction_idealamp = prediction.detach().clone()

            prediction_idealtime = prediction_idealtime.detach().cpu().squeeze()
            prediction_idealtime = (
                prediction_idealtime * self.config["scaling_max_value"]
            )

            noise_std_prediction_ext = noise_std_prediction_ext.detach().cpu().squeeze()
            noise_std_prediction_ext = (
                noise_std_prediction_ext * self.config["scaling_max_value"]
            )
            noise_std_prediction_ext = torch.clamp(noise_std_prediction_ext, min=0.75)
            gen_noise = torch.normal(mean=0, std=noise_std_prediction_ext)

            non_zero_mask = prediction != 0
            # Add the noise only to non-zero positions
            prediction[non_zero_mask] += gen_noise[non_zero_mask]

            prediction_noise = prediction.detach().clone()

            if self.save_valid_plots:
                generate_validation_plots(
                    self,
                    prediction_noise,
                    prediction_idealamp,
                    prediction_idealtime,
                    targets,
                    data,
                    log_dir=self._trainer.default_root_dir,
                )

        return prediction

    def predict_step(self, batch):
        read_id, data, *args = batch
        bs, seq_l = data.shape[:2]
        data = data.reshape(bs, seq_l, -1)

        enc_out, emb_out = self.encoders(data)

        noise_std_prediction = self.noise_sampler(emb_out)
        noise_std_prediction = noise_std_prediction[:, :, None]

        length_predict_out, _, _, noise_std_prediction_ext = self.length_regulator(
            emb_out=emb_out,
            x=enc_out,
            target=None,
            noise_std_prediction=noise_std_prediction,
            max_length=self.config["max_signal_len"],
            dwell_mean=self.dwell_mean,
            dwell_std=self.dwell_std,
            duration_sampling=self.duration_sampling,
        )


        prediction = self.decoders(length_predict_out)

        prediction = prediction * self.config["scaling_max_value"]
        prediction = prediction.squeeze(-1)
        
        if self.noise_std > 0:
            non_zero_mask = prediction != 0
            if self.noise_sampling:
                noise_std = noise_std_prediction_ext.squeeze() * self.noise_std * self.config["scaling_max_value"]

                gen_noise = torch.normal(mean=0, std=noise_std)

                prediction[non_zero_mask] += gen_noise[non_zero_mask]
            else:
                gen_noise = torch.normal(mean=0, std=self.noise_std, size=prediction.shape, device=prediction.device)

                prediction[non_zero_mask] += gen_noise[non_zero_mask]
        
        prediction = torch.clamp(prediction, min=0)

        d = {}
        for read, pred in zip(read_id, prediction):
            d.setdefault(read, []).append(pred)
        self.results.append(d)

        self.total_samples += data.shape[0]
        if isinstance(self.out_writer, BLOW5Writer) and self.total_samples >= self.export_every_n_samples:
            self.export_and_clear_results(keep_last=True)
            self.total_samples = 0  # Reset sample count
    

    def export_and_clear_results(self, keep_last: bool = True):
        """
        Export and clear the accumulated results.

        This method merges all accumulated result dictionaries into a single dictionary.
        Each entry in the result dictionary corresponds to a read ID and its associated predictions.
        The results are then saved using the `out_writer` and the self.results is cleared.

        Parameters
        ----------
        keep_last : bool, optional
            Whether to keep the last entry (i.e., the most recent read) if it may not be fully predicted.
            This is the case during prediction loop. Only at the last step, this option is deactivated.
            If `True`, the last read is preserved and appended back to the results after export.
            Default is `True`.
        """

        # Initialize a defaultdict to accumulate results from all batches
        res = defaultdict(list)
        # Merge all dicts into one
        for d in self.results:
            for k, v in d.items():
                res[k].extend(v)

        # Identify the last read and save it separatly as it may not be fully predicted
        last_read = None
        if keep_last and res:
            last_key = next(reversed(res))
            last_read = {last_key: res.pop(last_key)}

        # Concatenate all tensors associated with the same read ID
        for k, v in res.items():
            concatenated_tensor = torch.cat(v)
            res[k] = concatenated_tensor[concatenated_tensor.nonzero()].squeeze()

        self.out_writer.signals = res
        self.out_writer.save()
        self.out_writer.signals = []

        # Set results to empty list to clear memory
        self.results.clear()
        self.results = []

        # If there was a last read, re-add it back to results list
        if last_read:
            self.results.append(last_read)

        logger.debug("Results exported and memory cleared.")

    def on_predict_epoch_end(self):
        if self.results:
            self.export_and_clear_results(keep_last=False)
        logger.debug("Epoch end operation completed.")

    def configure_optimizers(self):
        if self.config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                eps=1e-7,
            )
        elif self.config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                eps=1e-7,
            )
        elif self.config["optimizer"] == "RAdam":
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "AdaFactor":
            optimizer = transformers.Adafactor(
                self.parameters(),
                relative_step=True,
                warmup_init=True,
                lr=None,
            )
            lr_scheduler = AdafactorSchedule(optimizer)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        elif self.config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "RMSProp":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

        num_warmup_steps = int(
            self.trainer.estimated_stepping_batches * self.config["warmup_ratio"]
        )

        if self.config["lr_schedule"] == "warmup_cosine":
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.config["lr_schedule"] == "warmup_constant":
            lr_scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif self.config["lr_schedule"] == "constant":
            lr_scheduler = transformers.get_constant_schedule(optimizer)
        elif self.config["lr_schedule"] == "warmup_cosine_restarts":
            lr_scheduler = (
                transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=self.trainer.estimated_stepping_batches,
                    num_cycles=2,
                )
            )
        elif self.config["lr_schedule"] == "one_cycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config["lr"],
                total_steps=self.trainer.estimated_stepping_batches,
            )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def log_total_gradients(self):
        """
        Log total gradient norm across all parameters.
        """
        non_empty_grads = [
            param.grad.norm().item()
            for param in self.parameters()
            if param.grad is not None
        ]
        if non_empty_grads:  # Check if there are non-empty gradients
            total_grad_norm = {
                "total_gradients": torch.norm(torch.tensor(non_empty_grads))
            }
            self.logger.log_metrics(total_grad_norm, step=self.global_step)

    def log_all_gradients(self):
        grad_norms = {
            name: torch.norm(param.grad).item()
            for name, param in self.named_parameters()
            if param.grad is not None
        }
        self.logger.log_metrics(grad_norms, step=self.global_step)

    def on_after_backward(self):
        # if self.config["gradient_logging"]:
        # log_interval = self.trainer.estimated_stepping_batches // 100
        # Log gradient norms
        if self.global_step % 100 == 0:  # Log every 100 steps
            # self.log_total_gradients()
            self.log_all_gradients()


def get_loss(
    self,
    prediction: torch.Tensor,
    targets: torch.Tensor,
    duration_pred_out: torch.Tensor,
    data_ls: torch.Tensor,
    dist_duration: torch.distributions.Distribution,
    noise_std_prediction: torch.Tensor,
    noise_std: torch.Tensor,
    step: str = "valid",
) -> torch.Tensor:
    """
    Computes the total loss given the prediction and targets. The total loss is the sum of the signal loss,
    duration loss, and noise standard deviation loss.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted signal values from the model.
    targets : torch.Tensor
        The ground truth signal values.
    duration_pred_out : torch.Tensor
        The predicted lengths of the signals.
    data_ls : torch.Tensor
        The ground truth signal lengths.
    dist_duration : torch.distributions.Distribution
        The distribution used to compute the duration loss.
    noise_std_prediction : torch.Tensor
        The predicted noise standard deviation.
    noise_std : torch.Tensor
        The ground truth noise standard deviation.
    step : str, optional
        The training step indicator (e.g., "train", "valid"). Default is "valid".

    Returns
    -------
    torch.Tensor
        The total computed loss.
    """

    # Signal Loss
    loss_func_signal = nn.MSELoss()
    signal_loss = loss_func_signal(prediction, targets).mean()
    self.log(f"{step}_signal_loss", signal_loss, prog_bar=True, sync_dist=True)

    # Duration Loss
    data_ls = torch.abs(data_ls) + (data_ls == 0).int()
    data_ls = data_ls.unsqueeze(-1)
    neg_log_likelihood = -dist_duration.log_prob(data_ls)
    duration_loss = torch.mean(neg_log_likelihood)
    duration_loss = duration_loss * 0.0005  # Scale duration loss down
    self.log(f"{step}_duration_loss", duration_loss, prog_bar=True, sync_dist=True)

    # Noise STD loss
    mse_loss = nn.MSELoss()
    noise_loss = mse_loss(noise_std.squeeze(), noise_std_prediction.squeeze())
    self.log(f"{step}_noise_loss", noise_loss, prog_bar=True, sync_dist=True)

    # Total Loss
    total_loss = signal_loss + duration_loss + noise_loss
    self.log(f"{step}_total_loss", total_loss, prog_bar=False, sync_dist=True)
    return total_loss
