#!/usr/bin/env python

"""
Preprocess training files to set of .npy files
Adapted from bonito https://github.com/nanoporetech/bonito
https://github.com/nanoporetech/bonito/blob/master/bonito/cli/convert.py
"""

import logging
import os
import numpy as np
from itertools import islice as take
from tqdm import tqdm
import polars as pl
from typing import List, Optional, Tuple, Generator, Any, Dict

from .utils import regular_break_points, one_hot_encode

logger = logging.getLogger("seq2squiggle")


class ChunkDataSet:
    """
    A dataset class for handling chunks of data for preprocessing. This class processes and stores
    chunks, targets, and associated length and standard deviation information.

    Attributes
    ----------
    chunks : np.ndarray
        The chunks of data, which are expanded along the second axis.
    targets : np.ndarray
        The target values associated with each chunk.
    c_lengths : np.ndarray
        The lengths of the chunks, expanded along the second axis.
    t_lengths : np.ndarray
        The target lengths corresponding to each chunk.
    stdevs : np.ndarray
        The standard deviations associated with each chunk, expanded along the second axis.

    Methods
    -------
    __getitem__(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple containing the chunk data, targets, chunk lengths, target lengths, and standard deviations for the given index.

    __len__() -> int
        Returns the number of items in the dataset.
    """

    def __init__(self, chunks, targets, c_lengths, t_lengths, stdevs):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.c_lengths = np.expand_dims(c_lengths, axis=1)
        self.t_lengths = t_lengths
        self.stdevs = np.expand_dims(stdevs, axis=1)

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.int32),
            self.targets[i].astype(np.float32),
            self.c_lengths[i].astype(np.int16),
            self.t_lengths[i].astype(np.int16),
            self.stdevs[i].astype(np.float32),
        )

    def __len__(self):
        return len(self.t_lengths)


def pad_lengths(
    ragged_signal: List[np.ndarray], max_len: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pads a ragged array (a list of arrays with different lengths) to a uniform length.

    Parameters
    ----------
    ragged_signal : List[np.ndarray]
        A list of numpy arrays with varying lengths.
    max_len : Optional[int]
        The maximum length to pad the arrays to. If None, the length of the longest array in `ragged_signal` is used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple where:
        - The first element is a 2D numpy array with padded arrays.
        - The second element is a 1D numpy array of the original lengths of the arrays in `ragged_signal`.
    """
    lengths = np.array([len(x) for x in ragged_signal], dtype=np.int16)

    padded_signal = np.zeros(
        (len(ragged_signal), max_len or np.max(lengths)), dtype=ragged_signal[0].dtype
    )

    for x, y in zip(ragged_signal, padded_signal):
        y[: len(x)] = x[: (max_len or np.max(lengths))]
    return padded_signal, lengths


def typical_indices(x, max_signal_len, n=2.5):
    """
    Returns indices of values in `x` that are within a certain range based on `max_signal_len` and standard deviations.

    Parameters
    ----------
    x : np.ndarray
        An array of numerical values.
    max_signal_len : int
        The maximum allowed length of signals. If `max_signal_len` is positive, the function returns indices where `x` values
        are within the range (0, `max_signal_len`].
    n : float, optional
        The number of standard deviations used to define the range around the mean for filtering values. Default is 2.5.

    Returns
    -------
    np.ndarray
        An array of indices where the condition is satisfied.
    """
    if max_signal_len <= 0:
        mu, sd = np.mean(x), np.std(x)
        (idx,) = np.where((mu - n * sd < x) & (x < mu + n * sd))
    else:
        (idx,) = np.where((0 < x) & (x <= max_signal_len))
    return idx


def filter_chunks(
    ds: "ChunkDataSet", idx: np.ndarray, max_value: int
) -> "ChunkDataSet":
    """
    Filters a ChunkDataSet instance to include only the data specified by indices and truncates targets.

    Parameters
    ----------
    ds : ChunkDataSet
        An instance of the ChunkDataSet class containing the data to be filtered.
    idx : np.ndarray
        An array of indices used to filter the data from the ChunkDataSet.
    max_value : int
        The maximum length to which targets should be truncated.

    Returns
    -------
    ChunkDataSet
        A new instance of ChunkDataSet with filtered and truncated data.
    """
    filtered = ChunkDataSet(
        ds.chunks.squeeze(1)[idx],
        ds.targets[idx],
        ds.c_lengths.squeeze(1)[idx],
        ds.t_lengths[idx],
        ds.stdevs[idx],
    )
    filtered.targets = filtered.targets[:, :max_value]
    return filtered


def load_numpy_datasets(
    path: str, limit: Optional[int] = None, directory: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads numpy arrays for chunks, targets, and lengths from a specified file path.

    Parameters
    ----------
    path : str
        Path to the .npz file containing the datasets.
    limit : Optional[int], default=None
        Maximum number of samples to load. If None, all samples are loaded.
    directory : Optional[str], default=None
        Directory path is not used in this function but kept for consistency with
        other functions that may use it.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing numpy arrays for chunks, targets, chunks lengths, targets lengths, and standard deviations.
    """

    with np.load(path, allow_pickle=False) as data:
        slices = slice(None, limit)
        return (
            np.array(data.get("chunks", [])[slices]),
            np.array(data.get("targets", [])[slices]),
            np.array(data.get("chunks_lengths", [])[slices]),
            np.array(data.get("targets_lengths", [])[slices]),
            np.array(data.get("stdevs", [])[slices]),
        )


def save_chunks(chunks, output_directory: str) -> None:
    """
    Saves the attributes of a ChunkDataSet instance to numpy files in the specified directory.

    Parameters
    ----------
    chunks : ChunkDataSet
        An instance of the ChunkDataSet class containing attributes to save.
    output_directory : str
        The directory where the numpy files will be saved.

    Returns
    -------
    None
    """
    os.makedirs(output_directory, exist_ok=True)

    # Define file names and their corresponding data attributes
    data_map = {
        "chunks": chunks.chunks.squeeze(1),
        "chunks_lengths": chunks.c_lengths.squeeze(1),
        "targets": chunks.targets,
        "targets_lengths": chunks.t_lengths,
        "stdevs": chunks.stdevs,
    }

    # Save and log each attribute
    for name, data in data_map.items():
        np.save(os.path.join(output_directory, f"{name}.npy"), data)
        logger.debug(f"  - {name}.npy with shape {data.shape}")

    logger.debug(f"> data written to: {output_directory}")


def save_chunks_in_batches(chunks, output_directory: str, counter: int = 0) -> None:
    """
    Saves the attributes of a ChunkDataSet instance to numpy files in batches, with filenames indicating the batch number.

    Parameters
    ----------
    chunks : ChunkDataSet
        An instance of the ChunkDataSet class containing attributes to save.
    output_directory : str
        The directory where the numpy files will be saved.
    counter : int, optional
        The batch number to include in the filenames, by default 0

    Returns
    -------
    None
    """
    os.makedirs(output_directory, exist_ok=True)

    data_map = {
        "chunks": chunks.chunks.squeeze(1),
        "chunks_lengths": chunks.c_lengths.squeeze(1),
        "targets": chunks.targets,
        "targets_lengths": chunks.t_lengths,
        "stdevs": chunks.stdevs,
    }

    for name, data in data_map.items():
        np.save(os.path.join(output_directory, f"{name}-{counter:04d}.npy"), data)
        logger.debug(f"  - {name}.npy with shape {data.shape}")

    logger.debug(f"> data written to: {output_directory}")


def get_chunks(
    chunks_in: Tuple[List[Any], List[Any], List[int], List[Tuple[int, int]], List[Any]],
    config: dict,
) -> Generator[Tuple[List[Any], List[Any], List[int], List[Any]], None, None]:
    """
    Returns a generator of tuples containing DNA chunks, corresponding signal chunks, and associated metadata.

    Parameters
    ----------
    chunks_in : tuple
        A tuple consisting of:
        - dna_seq: List of DNA sequences.
        - signal: List of signal values.
        - signal_len: List of signal lengths.
        - dna2signal_idxs: List of tuples indicating the start and end indices of DNA to signal mapping.
        - stdevs: List of standard deviations for the signals.
    config : dict
        Configuration dictionary containing:
        - "max_dna_len": Maximum length for DNA sequences.

    Returns
    -------
    Generator
        A generator yielding tuples of:
        - dna_chunk: List of DNA sequences.
        - signal_chunk: List of signal values corresponding to the DNA chunk.
        - signal_len_chunk: List of signal lengths for the DNA chunk.
        - stdev_chunk: List of standard deviations for the signal values.
    """
    dna_seq, signal, signal_len, dna2signal_idxs, stdevs = chunks_in

    breakpoints = regular_break_points(len(dna_seq), config["max_dna_len"])
    return (
        (
            dna_seq[i:j],
            signal[dna2signal_idxs[i][0] : dna2signal_idxs[j - 1][1]],
            signal_len[i:j],
            stdevs[i:j],
        )
        for (i, j) in breakpoints
    )


def read_slices(read_ids, df):
    """
    Yields slices of a DataFrame for each read ID provided.

    Parameters
    ----------
    read_ids : list of str
        List of read IDs to filter the DataFrame.
    df : pl.DataFrame
        Polars DataFrame from which to extract slices based on read IDs.

    Yields
    ------
    pl.DataFrame
        DataFrame slice filtered by the current read ID.
    """
    for read_id in read_ids:
        df_slice = df.filter(pl.col("read_name") == read_id)
        yield df_slice


def get_kmer(dna_seq: List[str], kmer_size: int) -> List[str]:
    """
    Generates k-mers of a specified size from a DNA sequence list.

    Parameters
    ----------
    dna_seq : list of str
        List of DNA sequences from which k-mers are to be extracted.
    kmer_size : int
        The size of the k-mer to generate. Must be between 3 and 9.

    Returns
    -------
    list of str
        List of k-mer sequences of the specified size.
    """
    if not (3 <= kmer_size <= 9):
        logger.error(f"Choose a kmer value between 3 and 9. You chose {kmer_size}")
        raise ValueError(f"Choose a kmer value between 3 and 9. You chose {kmer_size}")

    # Check the length of the first sequence
    seq_length = len(dna_seq[0])

    # for R9
    if seq_length == 6:
        slice_map = {6: slice(None), 5: slice(0, -1), 4: slice(1, -1), 3: slice(1, 4)}
    # for R10
    elif seq_length == 9:
        slice_map = {9: slice(None), 8: slice(1, None), 7: slice(1, -1), 6: slice(2, -1),
                     5: slice(3, -1), 4: slice(4, -1), 3: slice(5, -1)}
    else:
        logger.error("Sequence length should be 6 (R9.4) or 9 (R10.4).")
        raise ValueError("Sequence length should be 6 (R9.4) or 9 (R10.4).")

    if kmer_size > seq_length:
        logger.error(f"kmer_size {kmer_size} is larger than the sequence length {seq_length}.")
        raise ValueError(f"kmer_size {kmer_size} is larger than the sequence length {seq_length}.")

    return [seq[slice_map[kmer_size]] for seq in dna_seq]
    


def process_df(
    df: pl.DataFrame, config: dict
) -> Tuple[List[str], List[float], np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Processes a DataFrame to filter and transform DNA and signal data.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing columns for DNA sequences, signal data, and event standard deviations.
    config : dict
        Configuration dictionary containing parameters like maximum DNA length and k-mer size.

    Returns
    -------
    Tuple[List[str], List[float], np.ndarray, List[Tuple[int, int]], np.ndarray]
        - List of one-hot encoded DNA sequences.
        - Flattened list of signal values.
        - Array of signal lengths.
        - List of tuples representing start and end indices for DNA to signal mapping.
        - Array of standard deviations for each event.
    """
    # Filter out artifacts of uncalled4 signal processing
    df = df.sort(["position"]).filter(pl.col("model_kmer") != ("N" * config["seq_kmer"]))

    # add 0s so that remainder is 0
    df = df.with_columns(pl.col("end_idx").sub(pl.col("start_idx")).alias("signal_len"))
    # Filter out events above a length of 25.0
    df = df.filter(pl.col("signal_len") <= 25)
    # Filter out events above a noise of 10.0
    # df = df.filter(pl.col("event_stdv") <= 10)
    signal_len = df.select(pl.col(["signal_len"])).to_numpy().squeeze()

    # process DNA
    dna_seq = df["model_kmer"].to_list()    

    # Add remainder
    remain = config["max_dna_len"] - (len(dna_seq) % config["max_dna_len"])
    zero_array = [("_" * config["seq_kmer"])] * remain
    dna_seq = dna_seq + zero_array

    # One hot encode sequence
    dna_seq = one_hot_encode(dna_seq, len(dna_seq[0]))

    # Process the signal
    signal = df["samples"].to_list()
    signal = [i.split(",") for i in signal]

    # Process the noise std
    stdevs = df["event_stdv"].to_numpy()

    # Extend the list such that it has sam length as signal
    zero_array = np.zeros(remain, dtype=np.float32)
    # add 0s so that for each appended _ char you have a single signal point with value 0
    stdevs = np.append(stdevs, zero_array)

    # All signals into one list
    signal = [float(item) for sublist in signal for item in sublist]
    zero_array = np.zeros(remain, dtype=np.float32)

    # add 0s so that for each appended _ char you have a single signal point with value 0
    signal = np.append(signal, zero_array)

    one_array = np.ones(remain, dtype=np.int16)

    # add 1s so each _ maps to a 0 in the signal
    signal_len = np.append(signal_len, one_array)

    dna2signal_idxs = get_dna2signal_idxs(signal_len)
    return dna_seq, signal, signal_len, dna2signal_idxs, stdevs


def get_dna2signal_idxs(signal_len: List[int]) -> List[Tuple[int, int]]:
    """
    Generates a list of tuples representing the start and end indices for mapping DNA sequences to signal data.

    Parameters
    ----------
    signal_len : List[int]
        List of lengths of the signal segments corresponding to DNA sequences.

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples where each tuple contains the start and end indices for each DNA segment in the signal data.
    """
    start_idx = 0
    end_idx = 0
    dna2signal_idxs = []
    for i in signal_len:
        end_idx += i
        dna2signal_idxs.append((start_idx, end_idx))
        start_idx += i
    return dna2signal_idxs


def split_eventtable_in_chunks(
    event_df: Any, config: dict, num_chunks: Optional[int] = None
) -> ChunkDataSet:
    """
    Splits the event dataframe into chunks, processes each chunk, and pads them to a uniform length.

    Parameters
    ----------
    event_df : pd.DataFrame
        The dataframe containing event data to be processed. Must have columns "read_name" and other relevant data.
    config : dict
        Configuration dictionary containing parameters such as max_signal_len.
    num_chunks : Optional[int]
        The maximum number of chunks to process. If None, all chunks are processed.

    Returns
    -------
    ChunkDataSet
        An instance of ChunkDataSet containing the processed and padded chunks.
    """
    df_slice = event_df.partition_by("read_name")

    all_chunks = (
        (chunks, targets, chunk_len, stdevs)
        for df_single_read in df_slice
        for chunks, targets, chunk_len, stdevs in get_chunks(
            process_df(df_single_read, config), config
        )
    )

    chunks, targets, chunks_len, stdevs = zip(
        *tqdm(
            take(all_chunks, num_chunks), total=num_chunks, desc="Processing to chunks"
        )
    )

    logger.debug("Padding chunks")
    targets, targets_len = pad_lengths(targets, max_len=config["max_signal_len"])

    return ChunkDataSet(chunks, targets, chunks_len, targets_len, stdevs)


def process_events(
    events_path: str, outdir: str, max_chunks: int, config: Dict[str, any]
) -> None:
    """
    Reads, processes, filters, and saves event data from a TSV file.

    Parameters
    ----------
    events_path : str
        Path to the TSV file containing event data.
    outdir : str
        Output directory where processed chunks will be saved.
    max_chunks : int
        Maximum number of chunks to process.
    config : Dict[str, any]
        Configuration dictionary containing parameters like `max_signal_len`.

    Returns
    -------
    None
    """
    logger.debug("Reading and processing events.tsv")

    event_df = pl.read_csv(os.path.join(events_path), separator="\t")
    training_chunks = split_eventtable_in_chunks(event_df, config, max_chunks)
    logger.debug(
        f"  - Total amount of chunks: {training_chunks.chunks.squeeze(1).shape[0]}"
    )
    logger.debug("Filtering chunks")
    training_indices = typical_indices(
        training_chunks.t_lengths, config["max_signal_len"]
    )
    training_chunks = filter_chunks(
        training_chunks,
        np.random.permutation(training_indices),
        config["max_signal_len"],
    )
    logger.debug("Saving chunks")
    save_chunks(training_chunks, outdir)


def process_events_in_batches(
    events_path: str,
    outdir: str,
    max_chunks: int,
    chunksize: int,
    config: Dict[str, any],
) -> None:
    """
    Reads, processes, filters, and saves event data from a TSV file in batches.

    Parameters
    ----------
    events_path : str
        Path to the TSV file containing event data.
    outdir : str
        Output directory where processed chunks will be saved.
    max_chunks : int
        Maximum number of chunks to process.
    chunksize : int
        Number of rows to read per batch.
    config : Dict[str, any]
        Configuration dictionary containing parameters like `max_signal_len`.

    Returns
    -------
    None
        This function does not return a value but saves processed chunks to disk.
    """
    logger.debug("Read and process events.tsv in batches")

    reader = pl.read_csv_batched(
        events_path,
        separator="\t",
        batch_size=chunksize,
        n_rows=max_chunks,
    )
    batches = reader.next_batches(100)
    counter = 0
    while batches:
        logger.info(f"Processing batch {counter}")
        df_current_batches = pl.concat(batches)
        training_chunks = split_eventtable_in_chunks(df_current_batches, config)
        training_indices = typical_indices(
            training_chunks.t_lengths, config["max_signal_len"]
        )
        training_chunks = filter_chunks(
            training_chunks,
            np.random.permutation(training_indices),
            config["max_signal_len"],
        )
        save_chunks_in_batches(training_chunks, outdir, counter)
        counter += 1
        batches = reader.next_batches(100)


def preprocess_run(
    events_path: str, outdir: str, batches: bool, chunksize: int, config: Dict[str, any]
) -> None:
    """
    Preprocesses event data from a file by either processing it all at once or in batches.

    Parameters
    ----------
    events_path : str
        Path to the file containing event data. It can be a TSV or a compressed file.
    outdir : str
        Directory where the processed chunks will be saved.
    batches : bool
        Flag indicating whether to process data in batches (`True`) or not (`False`).
    chunksize : int
        Number of rows to read per batch. Used only if `batches` is `True`.
    config : Dict[str, any]
        Configuration dictionary containing parameters for processing, like `max_dna_len` and `max_chunks_train`.

    Returns
    -------
    None
    """
    max_chunks = config["max_dna_len"] * config["max_chunks_train"]
    if events_path.endswith(".gz"):
        logger.warning("Reading csv in batches is not supported for compressed files")
        logger.warning("File will be read at once.")
        batches = False

    if batches == False:
        process_events(events_path, outdir, max_chunks, config)
    else:
        process_events_in_batches(events_path, outdir, max_chunks, chunksize, config)
