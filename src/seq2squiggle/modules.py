#!/usr/bin/env python

"""
Basic modules of seq2squiggle model 
"""

import torch
from torch import nn
import torch.nn.functional as F
from numba import jit

from .layers import FFTBlock, get_sinusoid_encoding_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder module. Based on Feed Forward Transformer blocks.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_dna_len"]
        n_src_vocab = len(config["allowed_chars"]) * config["seq_kmer"]
        d_word_vec = config["dmodel"]
        d_model = config["dmodel"]
        n_head = config["encoder_heads"]
        n_layers = config["encoder_layers"]
        pre_layers = config["pre_layers"]
        self.pre_layers = pre_layers
        d_k = d_v = d_model // n_head
        d_inner = config["dff"]
        ff_dropout = config["encoder_dropout"]
        att_dropout = config["encoder_dropout"]
        self.max_seq_len = n_position
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.src_emb = nn.Linear(n_src_vocab, d_model)
        self.relu = nn.ReLU()
        self.pre_net_stack = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(pre_layers)]
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    ff_dropout=ff_dropout,
                    att_dropout=att_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, return_attns=False):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        enc_slf_attn_list = []

        src_seq = self.src_emb(src_seq)
        src_seq = self.relu(src_seq)
        if self.pre_layers > 0:
            for pre_layer in self.pre_net_stack:
                src_seq = pre_layer(src_seq)
                src_seq = self.relu(src_seq)

        enc_output = src_seq + get_sinusoid_encoding_table(
            src_seq.shape[1], src_seq.shape[2]
        )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
            src_seq.device
        )

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=None, slf_attn_mask=None
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, src_seq


class Decoder(nn.Module):
    """
    Decoder module. Based on Feed Forward Transformer blocks.
    """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_signal_len"]
        n_layers = config["decoder_layers"]
        d_word_vec = config["dmodel"]
        d_model = config["dmodel"]
        n_head = config["decoder_heads"]
        ff_dropout = config["decoder_dropout"]
        att_dropout = config["decoder_dropout"]
        d_k = d_v = d_model // n_head
        d_inner = config["dff"]
        self.max_seq_len = n_position
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.out_linear = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.layer_stack_FFT = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    ff_dropout=ff_dropout,
                    att_dropout=att_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for dec_layer in self.layer_stack_FFT:
            dec_output, _ = dec_layer(dec_output, mask=None, slf_attn_mask=None)
        dec_output = self.out_linear(dec_output)
        dec_output = self.relu(dec_output)
        return dec_output


class DurationSampler(nn.Module):
    """
    Duration sampler for sampling the length of a signal given the encoder output.

    The `DurationSampler` uses a neural network to predict the parameters of a Gamma distribution, which is
    then used to sample the duration of a signal.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters. Expected keys are:
        - "dmodel": Dimensionality of the model's output.
        - "duration_dropout": Dropout rate to be applied in the network.

    Attributes
    ----------
    d : int
        Dimensionality of the model's output.
    output_size : int
        Size of the output for the linear layers.
    dropout : float
        Dropout rate.
    config : dict
        Configuration dictionary.
    conc_layer : nn.Sequential
        Sequential neural network layer for predicting the concentration parameter of the Gamma distribution.
    rate_layer : nn.Sequential
        Sequential neural network layer for predicting the rate parameter of the Gamma distribution.
    """

    def __init__(self, config):
        super(DurationSampler, self).__init__()
        self.d = config["dmodel"]
        self.output_size = 1
        self.dropout = config["duration_dropout"]
        self.config = config

        self.conc_layer = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.output_size),
            nn.Softplus(),  # enforces positivity
        )
        self.rate_layer = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.output_size),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, encoder_output: torch.Tensor):
        """
        Forward pass of the DurationSampler.

        Parameters
        ----------
        encoder_output : torch.Tensor
            The output tensor from the encoder, used to predict the Gamma distribution parameters.

        Returns
        -------
        Tuple[torch.Tensor, torch.distributions.Gamma]
            - out : torch.Tensor
                The sampled durations from the Gamma distribution, clamped to a minimum of 1.0.
            - dist : torch.distributions.Gamma
                The Gamma distribution object parameterized by the concentration and rate.
        """
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        conc = self.conc_layer(encoder_output)
        conc = torch.clamp(conc, min=epsilon)
        rate = self.rate_layer(encoder_output)
        rate = torch.clamp(rate, min=epsilon)

        dist = torch.distributions.gamma.Gamma(concentration=conc, rate=rate)
        out = dist.sample()
        out = torch.clamp(out, min=1.0)
        out = out.flatten(1)
        return out, dist


class NoiseSampler(nn.Module):
    """
    Noise sampler for generating noise values given the encoder output.

    The `NoiseSampler` uses a neural network to predict the standard deviation of the noise distribution
    based on the encoder's output. This standard deviation is then used to sample noise for the signal.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters. Expected keys are:
        - "max_dna_len": Maximum length of DNA sequences (used for input size, but not directly in this class).
        - "dmodel": Dimensionality of the model's output.
        - "duration_dropout": Dropout rate to be applied in the network.

    Attributes
    ----------
    input_size : int
        The size of the input data (in this case, the maximum length of DNA sequences).
    d : int
        Dimensionality of the model's output.
    output_size : int
        Size of the output for the linear layers.
    dropout : float
        Dropout rate.
    config : dict
        Configuration dictionary.
    stdv_layer : nn.Sequential
        Sequential neural network layer for predicting the standard deviation of the noise distribution.
    """

    def __init__(self, config):
        super(NoiseSampler, self).__init__()
        self.input_size = config["max_dna_len"]
        self.d = config["dmodel"]
        self.output_size = 1
        self.dropout = config["duration_dropout"]
        self.config = config

        self.stdv_layer = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.output_size),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, x):
        stdv = self.stdv_layer(x)
        stdv = stdv.flatten(1)
        return stdv


@jit(nopython=True)
def create_alignment(base_mat, duration_predictor_output):
    """
    Create an alignment matrix based on the duration predictions.

    This function fills a base matrix with alignment values based on the
    durations predicted for each sequence. For each predicted duration,
    the function updates the corresponding positions in the base matrix.

    Parameters
    ----------
    base_mat : np.ndarray
        The base matrix to be filled with alignment values. It should have
        shape (N, max_duration, L) where N is the number of sequences,
        max_duration is the maximum duration predicted, and L is the length of each sequence.

    duration_predictor_output : np.ndarray
        The predicted durations for each sequence. It should have shape (N, L) where
        N is the number of sequences and L is the length of each sequence. Each element
        represents the predicted duration for the corresponding position in the sequence.

    Returns
    -------
    np.ndarray
        The updated base matrix with alignment values filled in based on the predicted durations.
    """
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count + k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """
    Length regulator. Duplicates hidden states of encoder output to achieve
    alignment between DNA input and target signal data.

    Attributes
    ----------
    config : dict
        Configuration dictionary containing model parameters.
    duration_sampler : DurationSampler
        An instance of the DurationSampler class used to sample durations.

    Methods
    -------
    LR(x: torch.Tensor, duration_pred_out: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor
        Regulates the length of the input tensor x based on the duration predictions.

    forward(
        emb_out: torch.Tensor,
        x: torch.Tensor,
        noise_std_prediction: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        target: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        ideal_length: float = 0.0,
        duration_sampling: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.distributions.Distribution], Optional[torch.Tensor]]
        Processes the input tensors and applies length regulation based on duration predictions.
    """

    def __init__(self, config):
        super(LengthRegulator, self).__init__()
        self.config = config
        self.duration_sampler = DurationSampler(self.config)

    def LR(self, x, duration_pred_out, max_length=None):
        """
        Regulates the length of the input tensor x based on the duration predictions.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, max_duration, dna_length).
        duration_pred_out : torch.Tensor
            The tensor with predicted durations, shape (batch_size, dna_length).
        max_length : Optional[int]
            The maximum length to pad the output tensor.

        Returns
        -------
        torch.Tensor
            The length-regulated tensor with the same shape as x.
        """
        # largest value
        expand_max_len = torch.max(torch.sum(duration_pred_out, -1), -1)[0]
        # intialize array filled with 0s, shape (bs, expand_max_len, dna_length)
        alignment = torch.zeros(
            duration_pred_out.size(0), expand_max_len, duration_pred_out.size(1)
        ).numpy()
        # create alignment matrix
        alignment = create_alignment(alignment, duration_pred_out.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(device)
        # matrix multp
        output = alignment @ x
        # pad to max length
        if max_length:
            output = F.pad(output, (0, 0, 0, max_length - output.size(1), 0, 0))
        return output

    def forward(
        self,
        emb_out,
        x,
        noise_std_prediction=None,
        alpha=1.0,
        target=None,
        max_length=None,
        ideal_length=0.0,
        duration_sampling=True,
    ):
        min_value = 3
        if duration_sampling:
            duration_predictor_output, dist = self.duration_sampler(
                emb_out.detach().clone()
            )

            duration_predictor_output = torch.clamp(
                duration_predictor_output, min=min_value
            )
        else:
            bs, seq, _ = emb_out.shape
            dist = None
            if ideal_length > 0:
                duration_predictor_output = torch.full((bs, seq), ideal_length).to(
                    device
                )
            else:
                mean = torch.full((bs, seq), 9.0).to(device)
                std = torch.full((bs, seq), 4.0).to(device)
                duration_predictor_output = torch.normal(mean=mean, std=std)
                # Ensure all values are positive by clipping to a minimum value
                # a small positive value to ensure strictly positive values
                duration_predictor_output = torch.clamp(
                    duration_predictor_output, min=min_value
                )

        if target is not None:
            output = self.LR(x, target, max_length=max_length)
            if noise_std_prediction is not None:
                noise_std_prediction = self.LR(
                    noise_std_prediction, target, max_length=max_length
                )
        else:
            duration_prediction = duration_predictor_output.detach().clone()
            duration_prediction = torch.round(duration_prediction).int()
            output = self.LR(x, duration_prediction, max_length=max_length)
            if noise_std_prediction is not None:
                noise_std_prediction = self.LR(
                    noise_std_prediction, duration_prediction, max_length=max_length
                )
        return output, duration_predictor_output, dist, noise_std_prediction
