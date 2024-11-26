#!/usr/bin/env python

"""
Different functions
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import logging
import torch
import sys
from pysam import FastxFile
import re
import scipy.stats as st
from prettytable import PrettyTable
from matplotlib.ticker import AutoMinorLocator
import platform
import psutil
import multiprocessing
from typing import List, Generator, Tuple, Union
from uuid import uuid4

logger = logging.getLogger("seq2squiggle")


def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return n_cpu // n_gpu if (n_gpu := torch.cuda.device_count()) > 1 else n_cpu


def one_hot_encode(sequences: List[str], seq_len: int) -> np.ndarray:
    """
    One-hot encodes a list of DNA sequences.

    Parameters
    ----------
    sequences : list of str
        A list where each string is a DNA sequence containing characters from {"_", "A", "C", "G", "T"}.
    seq_len: int
        Length of the input k-mer sequences

    Returns
    -------
    np.ndarray
        A 3D numpy array of shape (n_sequences, sequence_length, n_letters), where n_sequences is the number of sequences,
        sequence_length is the length of each sequence (assumed to be 9), and n_letters is the number of unique letters (5).
    """
    # Define the mapping of letters to integers
    letter_to_int = {"_": 0, "A": 1, "C": 2, "G": 3, "T": 4}

    # Determine the number of unique letters
    n_letters = len(letter_to_int)

    # Initialize an empty array to store the one-hot encoded sequences
    n_outer_sequences = len(sequences)
    one_hot_encoded = np.zeros((n_outer_sequences, seq_len, n_letters), dtype=np.float16)

    # Iterate through each outer sequence and its inner sequences, and one-hot encode them
    for i, outer_sequence in enumerate(sequences):
        for k, letter in enumerate(outer_sequence):
            if letter in letter_to_int:
                one_hot_encoded[i, k, letter_to_int[letter]] = 1

    return one_hot_encoded


def decode_one_hot(one_hot_encoded: np.ndarray) -> List[str]:
    """
    Decodes a one-hot encoded numpy array back to DNA sequences.

    Parameters
    ----------
    one_hot_encoded : np.ndarray
        A 3D numpy array of shape (n_sequences, sequence_length, n_letters), where n_sequences is the number of sequences,
        sequence_length is the length of each sequence, and n_letters is the number of unique letters.

    Returns
    -------
    List[str]
        A list of strings, where each string is a decoded DNA sequence.
    """
    # Define the reverse mapping of integers to letters
    int_to_letter = {0: "_", 1: "A", 2: "C", 3: "G", 4: "T"}

    # Determine the number of letters
    n_letters = len(int_to_letter)

    # Initialize an empty list to store the decoded sequences
    decoded_sequences = []

    # Iterate through each one-hot encoded sequence and decode it
    for sequence in one_hot_encoded:
        decoded_sequence = ""
        for letter_one_hot in sequence:
            # Find the index of the one-hot encoded letter
            letter_index = int(np.argmax(letter_one_hot))
            # Map the index to the corresponding letter
            decoded_sequence += int_to_letter[letter_index]
        decoded_sequences.append(decoded_sequence)

    return decoded_sequences


def get_profile(profile):
    """
    Get the profile dict for generating a pod5/slow5 file

    The profile includes the digitisation, sample_rate, range, offset_mean, offset_std, median_before_mean, median_before_std

    -------
    Arguments
    string
        Name of the profile

    -------

    Returns

    dict
        Your profile

    -------
    """
    profiles = {
        "dna-r10-min": {
            "digitisation": 8192,
            "sample_rate": 5000,
            "range": 1536.598389,
            "offset_mean": 13.380569389019,
            "offset_std": 16.311471649012,
            "median_before_mean": 202.15407438804,
            "median_before_std": 13.406139241768,
        },
        "dna-r10-prom": {
            "digitisation": 2048,
            "sample_rate": 5000,
            "range": 281.345551,
            "offset_mean": -127.5655735,
            "offset_std": 19.377283387665,
            "median_before_mean": 189.87607393756,
            "median_before_std": 15.788097978713,
        },
        "dna-r9-min": {
            "digitisation": 8192,
            "sample_rate": 4000,
            "range": 1443.030273,
            "offset_mean": 13.7222605,
            "offset_std": 10.25279688,
            "median_before_mean": 200.815801,
            "median_before_std": 20.48933762,
        },
        "dna-r9-prom": {
            "digitisation": 2048,
            "sample_rate": 4000,
            "range": 748.5801,
            "offset_mean": -237.4102,
            "offset_std": 14.1575,
            "median_before_mean": 214.2890337,
            "median_before_std": 18.0127916,
        },
    }

    if profile in profiles:
        return profiles[profile]
    else:
        logger.error(f"Incorrect value for profile: {profile}")


def update_profile(profile_dict, **kwargs):
    """
    Update the profile dictionary with the provided parameters.

    Any parameter in kwargs that is not None will replace the corresponding 
    value in the profile_dict.

    -------
    Arguments
    dict
        The current profile dictionary to update
    kwargs
        The parameters to update in the profile dictionary

    -------
    Returns
    dict
        The updated profile dictionary
    """
    for key, value in kwargs.items():
        if value is not None and key in profile_dict:
            profile_dict[key] = value
        elif key not in profile_dict:
            logger.warning(f"Warning: {key} is not a valid key in the profile")
    
    return profile_dict

def update_config(profile_name, config):
    """
    Updates the configuration dictionary with the appropriate sequence k-mer size
    based on the profile name.

    Parameters:
        profile_name (str): The profile name, typically indicating sequencing chemistry.
        config (dict): The configuration dictionary to update.

    Returns:
        dict: The updated configuration dictionary.
    """
    if profile_name.startswith("dna-r10"):
        config["seq_kmer"] = 9
    elif profile_name.startswith("dna-r9"):
        config["seq_kmer"] = 6
    else:
        raise ValueError(f"Unsupported profile name: {profile_name}. Expected 'dna-r10' or 'dna-r9' prefix.")
    return config


def regular_break_points(n, chunk_len, overlap=0, align="left"):
    """
    Returns breakpoints of a signal given a chunk_len.
    Depending on the alignment parameter it will either remove remainders at different positions

    -------
    Arguments:
        n  - Raw signal
        chunk_len - lenght of desired chunks, int
        overlap - desired overlap, positive int
        align - Alignment option, must be "left", "mid", or "right"
    -------
    Returns
        breakpoints
    """
    num_chunks, remainder = divmod(n - overlap, chunk_len - overlap)

    start = {"left": 0, "mid": remainder // 2, "right": remainder}[align]
    starts = np.arange(
        start, start + num_chunks * (chunk_len - overlap), (chunk_len - overlap)
    )
    return np.vstack([starts, starts + chunk_len]).T


def read_fasta(path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Reads sequences and their names from a FASTA file.

    Parameters
    ----------
    path : str
        Path to the FASTA file.

    Yields
    ------
    Generator[Tuple[str, str], None, None]
        A generator that yields tuples, where each tuple contains:
        - sequence (str): The nucleotide sequence.
        - name (str): The identifier of the sequence.
    """
    with FastxFile(path) as fh:
        for entry in fh:
            yield (entry.sequence, entry.name)


def draw_gamma_dis(mean, seed, total_len):
    sample = st.gamma.rvs(6.3693711, 0.53834893, size=1, random_state=seed)
    sample = int(sample * mean / 4.39)
    sample = np.clip(sample, 1, total_len)
    return sample


def draw_beta_dis(mean, seed, total_len):
    sample = st.beta.rvs(1.778, 7.892, 316.758, 34191.257, size=1, random_state=seed)
    sample = (sample[0] * mean / 6615.0).astype(int)
    sample = np.clip(sample, 1, total_len)
    return sample


def draw_expon_dis(mean, seed, total_len):
    sample = st.expon.rvs(
        loc=213.98910256668592, scale=6972.5319847131141, size=1, random_state=seed
    )
    sample = (sample[0] * mean / 7106.0).astype(int)
    sample = np.clip(sample, 1, total_len)
    return sample


def extract_kmers(dna_string, k):
    kmers = []
    n = len(dna_string)
    for i in range(n - k + 1):
        kmers.append(dna_string[i : i + k])
    return kmers


def add_remainder(x, max_dna, k):
    remain = max_dna - (len(x) % max_dna)
    if remain % max_dna > 0:
        zero_array = [("_" * k)] * remain
        x += zero_array
    return x


def split_sequence(x, config):
    x = extract_kmers(x, config["seq_kmer"])
    x = add_remainder(x, config["max_dna_len"], config["seq_kmer"])
    x = one_hot_encode(x, config["seq_kmer"])
    breakpoints = regular_break_points(len(x), config["max_dna_len"], align="left")
    x_breaks = np.array([x[i:j] for (i, j) in breakpoints])
    return x_breaks


def get_genome_and_position(genome_lengths, random_position):
    total_length = sum(genome_lengths)
    if random_position >= total_length:
        raise ValueError("Random position exceeds the total length of genomes")

    cumulative_length = 0
    for i, length in enumerate(genome_lengths):
        cumulative_length += length
        if random_position < cumulative_length:
            genome_index = i
            position_within_genome = random_position - (cumulative_length - length)
            return genome_index, position_within_genome


def sample_single_genome(genome, read_length, start_index):
    """
    Sample a read of specified length from a single genome.
    """
    end_index = start_index + read_length
    return genome[start_index:end_index]


def read_check(read, read_length, read_i):
    if len(read) != read_length:
        logger.debug(
            f"Sampled Read length of read {read_i} is shorter than real read length."
        )
        return False
    if len(read) < 100:
        logger.debug(f"Sampled Read length of read {read_i} is too short.")
        return False
    count_N = read.count("N")
    if count_N > 0.1 * read_length:
        logger.debug(
            f"Too many 'N' bases ({count_N} out of {read_length}) for read {read_i}"
        )
        return False

    return True


def N_to_ACTG(read):
    return "".join(random.choice("ACGT") if base == "N" else base for base in read)


def get_strand():
    return random.choice("+-")


def reverse_complement(f):
    complement_dict = {"A": "T", "T": "A", "C": "G", "G": "C"}
    r = [complement_dict.get(base, base) for base in reversed(f)]
    return "".join(r)


def sampling(
    num_seqs, genome_seqs, genome_lens, r, seed, total_len, distr, max_retries=20
):
    """
    Sample reads from each single genome.
    """

    distr_funcs = {
        "beta": draw_beta_dis,
        "gamma": draw_gamma_dis,
        "expon": draw_expon_dis,
    }

    sampled_reads = []

    total_genome_len = sum(genome_lens)

    reads = list(range(num_seqs))

    for read_i in reads:
        retries = 0
        while retries < max_retries:
            # Find the corresponding genome and its start index
            start_pos = random.randint(0, total_genome_len - 1)
            genome_index, start_index = get_genome_and_position(genome_lens, start_pos)
            genome = genome_seqs[genome_index]

            # Generate a unique seed for each read and retry combination
            unique_seed = seed + read_i * (max_retries + 1) + retries
            read_length = distr_funcs[distr](r, unique_seed, total_len)

            read = sample_single_genome(genome, read_length, start_index)

            read_strand = get_strand()

            if read_check(read, read_length, read_i):
                if "N" in read:
                    read = N_to_ACTG(read)

                if read_strand == "-":
                    read = reverse_complement(read)

                sampled_reads.append(read)
                break
            else:
                retries += 1
    return sampled_reads


def export_fasta(read_l, fasta):
    file_name, _ = os.path.splitext(fasta)
    out_file = f"{file_name}_reads.fasta"
    with open(out_file, "w") as f:
        for i, read in enumerate(read_l):
            f.write(f"{str(uuid4())}\n{''.join(read)}\n")
    return out_file


def yield_reads(reads):
    return ((read, str(uuid4())) for i, read in enumerate(reads))


def sample_reads_from_genome(
    genome_seqs: List[str],
    genome_lens: List[int],
    n: int,
    r: int,
    c: int,
    config: dict,
    fasta: str,
    seed: int,
    save: bool = False,
    distr: str = "expon"
) -> Tuple[Union[str, None], int]:
    """
    Sample reads from genome sequences based on the provided parameters.

    Parameters
    ----------
    genome_seqs : List[str]
        List of genome sequences.
    genome_lens : List[int]
        List of lengths of genome sequences.
    n : int
        Number of reads to generate. Use -1 to ignore and use coverage instead.
    r : int
        Minimum read length.
    c : int
        Coverage of the genome. Use -1 to ignore and use number of reads instead.
    config : dict
        Configuration dictionary containing parameters like "max_dna_len".
    fasta : str
        Path to the FASTA file where reads will be saved if `save` is True.
    seed : int
        Random seed for reproducibility.
    save : bool, optional
        If True, save the generated reads to a FASTA file (default is False).
    distr : str, optional
        Distribution type for sampling reads (default is "expon").

    Returns
    -------
    Tuple[Union[str, None], int]
        If `save` is True, returns a tuple of the path to the saved reads FASTA file and the total length of reads.
        If `save` is False, returns a tuple of a generator for the reads and the total length of reads.
    """
    logger.debug("Generating reads.")
    if n == -1 and c == 1:
        logger.error("You need to specify the coverage c or the number of reads n")
        raise ValueError("You need to specify the coverage c or the number of reads n")

    if n != -1 and c != -1:
        logger.error(
            "You can only either specify the coverage c or the number of reads, but not both"
        )
        raise ValueError(
            "You can only either specify the coverage c or the number of reads, but not both"
        )

    total_len = sum([len(seq) for seq in genome_seqs])
    seq_num = round(c * total_len / r)
    seq_num = n if n != -1 else seq_num
    logger.debug(f"Number of reads: {seq_num}")

    read_list = sampling(seq_num, genome_seqs, genome_lens, r, seed, total_len, distr)

    total_l = sum([round(len(read) / config["max_dna_len"]) for read in read_list])

    if save:
        reads_fasta = export_fasta(read_list, fasta)
    else:
        reads_fasta = yield_reads(read_list)

    logger.debug("Generating reads finished.")

    return reads_fasta, total_l


def load_genome(fasta):
    with FastxFile(fasta) as fh:
        for entry in fh:
            yield str(entry.sequence)


def process_genome(genome_seq: str):
    genome_seq = genome_seq.upper()
    genome_seq = re.sub(r"[^ATCG]", "N", genome_seq)
    return genome_seq, (len(genome_seq))


def compute_totals(generator):
    total_reads = 0
    total_length = 0
    for sequence, _ in generator:
        total_reads += 1
        total_length += len(sequence)
    return total_reads, total_length


def preprocess_genome(fasta: str):
    """
    Preprocessing the genome sequence


    -------
    Arguments

    Genome sequence [str]

    -------
    Returns

    Processed genome sequence [str]

    -------
    """

    logger.debug("Preprocessing the genome")
    # Append all genome seqs
    genome_seqs = load_genome(fasta)

    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_genome, genome_seqs)

    genome_seq_list, genome_len_list = zip(*results)

    logger.debug("Preprocessing the genome finished.")
    return genome_seq_list, genome_len_list


def get_reads(fasta, read_input, n, r, c, config, distr, seed, save=False):
    if read_input == False:
        logger.info(f"Genome mode. Reads will be generated from input fasta: {fasta}")
        genome_seqs, genome_lens = preprocess_genome(fasta)
        reads_fasta, total_l = sample_reads_from_genome(
            genome_seqs, genome_lens, n, r, c, config, fasta, seed, save, distr
        )
        if save:
            return read_fasta(reads_fasta)
        return reads_fasta, total_l
    else:
        logger.info(
            f"Read mode. Signals will be simulated directly from reads: {fasta}"
        )
        reads_generator = read_fasta(fasta)
        total_reads, total_length = compute_totals(reads_generator)
        # Recreate the generator for actual use
        reads_generator = read_fasta(fasta)
        return reads_generator, total_length


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def setup_logging(verbosity):

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console:
    console_formatter = logging.Formatter(
        "{name} {levelname} {asctime}: {message}", style="{", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)

    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def set_seeds(seed):
    """
    Sets random seeds using numpy, torch, random, and os given the seed in the config.

    Arguments:
    seed - int seed value
    """
    logger.info(f"Setting seeds using random seed {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def onehot2dna(seq_input, allowed_chars):
    """
    Return list of DNA sequence given one hot encoded list

    Arguments:
    seq_input - One hot encoded DNA sequence, must be list, containing only 0123
    """
    mapping = dict(zip(range(len(allowed_chars)), allowed_chars))
    seq_out = [mapping[i] for i in seq_input]
    return seq_out


def dna2onehot(seq_input, allowed_chars):
    """
    Return list of one hot encoded DNA sequence given DNA list

    Arguments:
    seq_input - DNA sequence, must be list, containing only ACGT
    """
    mapping = dict(zip(allowed_chars, range(len(allowed_chars))))
    seq_out = [mapping[i] for i in seq_input]
    return seq_out


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def generate_validation_plots(
    self,
    prediction,
    prediction_idealamp,
    prediction_idealtime,
    targets,
    data,
    log_dir,
    bs=12,
    devices=0,
):
    # Adjust batch size if it exceeds the number of targets
    bs = min(bs, targets.shape[0])

    # Select a subset of data for plotting
    targets = targets[:bs]
    prediction = prediction[:bs]
    prediction_idealamp = prediction_idealamp[:bs]
    prediction_idealtime = prediction_idealtime[:bs]
    data = data[:bs]

    for batch_idx, (
        batch_pred,
        batch_pred_idealamp,
        batch_pred_idealtime,
        batch_target,
        batch_dna,
    ) in enumerate(
        zip(prediction, prediction_idealamp, prediction_idealtime, targets, data)
    ):
        plot_signal(
            batch_pred,
            batch_pred_idealamp,
            batch_pred_idealtime,
            batch_target,
            batch_dna,
            self.current_epoch,
            self.config["log_name"],
            batch_idx,
            log_dir=log_dir,
        )


def plot_signal(
    batch_pred,
    batch_pred_idealamp,
    batch_pred_idealtime,
    target,
    data_input,
    epoch,
    log_name,
    batch_idx,
    log_dir,
    dec_output_FR=None,
):
    # Convert data_input to numpy array
    data_input = data_input.cpu().data.numpy()

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Set labels and grid
    plt.xlabel("Time")
    plt.ylabel("Normalized signal")
    plt.grid(which="major", linestyle="solid")
    plt.grid(which="minor", linestyle=(0, (1, 10)), axis="y")

    # Plot true and predicted signals
    X_time = range(0, len(target.flatten()))
    plt.plot(X_time, target.flatten(), label="True")
    plt.plot(X_time, batch_pred.flatten(), label="Default pred")
    plt.plot(X_time, batch_pred_idealamp.flatten(), label="Ideal amp pred")
    plt.plot(X_time, batch_pred_idealtime.flatten(), label="Ideal time pred")
    plt.legend(loc="lower left")

    # Create output directory if it doesn't exist
    out_dir = os.path.join(log_dir, f"epoch_{epoch}/")
    os.makedirs(out_dir, exist_ok=True)

    # Save plot as PNG file
    out_file = os.path.join(out_dir, f"batch_{batch_idx}.png")
    plt.savefig(out_file, dpi=100)

    # Clear plot and close figure to release memory
    plt.clf()
    plt.close()
