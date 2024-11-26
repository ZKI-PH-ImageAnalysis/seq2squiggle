#!/usr/bin/env python

"""
Dataloader and DataGenerator for training and prediction
"""

import os
import numpy as np
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset, Dataset, get_worker_info, DistributedSampler
from torch.distributed import init_process_group, get_rank, get_world_size
from multiprocessing.pool import ThreadPool as Pool
import itertools
import multiprocessing
import torch.distributed as tdi

from typing import Tuple, List, Optional, Dict, Generator
from bisect import bisect
from sklearn.model_selection import train_test_split

from .utils import split_sequence

logger = logging.getLogger("seq2squiggle")


class PoreDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning `DataModule` for managing data loading for training, validation, and prediction.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing settings for data processing and loading.
    total_l : int, optional
        The total length of the dataset. Default is 1.
    data_dir : str, optional
        Path to the directory containing training and prediction data. Default is "path/to/dir".
    valid_dir : str, optional
        Path to the directory containing validation data. Default is "path/to/dir".
    batch_size : int, optional
        Number of samples per batch. Default is 128.
    n_workers : int, optional
        Number of worker processes for data loading. Default is 1.

    Attributes
    ----------
    data_dir : str
        Path to the directory containing training and prediction data.
    valid_dir : str
        Path to the directory containing validation data.
    batch_size : int
        Number of samples per batch.
    config : dict
        Configuration dictionary used for data processing and loading.
    n_workers : int
        Number of worker processes for data loading.
    total_l : int
        Total length of the dataset.
    train_loader_kwargs : dict
        Keyword arguments for creating the training DataLoader.
    valid_loader_kwargs : dict
        Keyword arguments for creating the validation DataLoader.
    predict_loader_kwargs : dict
        Keyword arguments for creating the prediction DataLoader.

    Methods
    -------
    setup(stage: Optional[str] = None)
        Prepares the data loaders based on the provided stage (training, validation, or prediction).
    train_dataloader() -> DataLoader
        Returns the DataLoader for the training dataset.
    val_dataloader() -> DataLoader
        Returns the DataLoader for the validation dataset.
    predict_dataloader() -> DataLoader
        Returns the DataLoader for the prediction dataset.
    """

    def __init__(
        self,
        config,
        total_l: int = 1,
        data_dir: str = "path/to/dir",
        valid_dir: str = "path/to/dir",
        batch_size: int = 128,
        n_workers: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.config = config
        self.n_workers = n_workers
        self.total_l = total_l
        self.rank = rank
        self.world_size = world_size

    def setup(self, stage: str):
        if stage in ("fit", "validate"):
            (
                self.train_loader_kwargs,
                self.valid_loader_kwargs,
            ) = load_numpy(
                train_limit=self.config["max_chunks_train"],
                valid_limit=self.config["max_chunks_valid"],
                npy_train=self.data_dir,
                npy_valid=self.valid_dir,
                train_valid_split=self.config["train_valid_split"],
                config=self.config,
                # valid will be split from train if none is given
            )
        if stage in (None, "predict"):
            logger.debug("Loading fasta started")
            self.predict_loader_kwargs = load_fasta(
                self.data_dir, self.config, self.total_l, self.rank, self.world_size
            )
            logger.debug("Loading fasta ended")

    def train_dataloader(self):
        train_loader = DataLoader(
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            **self.train_loader_kwargs,
        )
        logger.info(f"True Training dataset size: {len(train_loader.dataset)}")
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            **self.valid_loader_kwargs,
        )
        logger.info(f"True Validation dataset size {len(valid_loader.dataset)}")
        return valid_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            pin_memory=True,
            **self.predict_loader_kwargs,
        )
        logger.info(f"True Prediction dataset size {len(predict_loader.dataset)}")
        return predict_loader


class ChunkDataSetMemmap(Dataset):
    """
    A PyTorch `Dataset` that uses memory-mapped files to efficiently handle large datasets for training and validation.

    Parameters
    ----------
    chunks_path : list of str
        List of file paths for chunk data stored in memory-mapped files.
    targets_path : list of str
        List of file paths for target data stored in memory-mapped files.
    c_lengths_path : list of str
        List of file paths for chunk lengths data stored in memory-mapped files.
    t_lengths_path : list of str
        List of file paths for target lengths data stored in memory-mapped files.
    stdevs_path : list of str
        List of file paths for standard deviations data stored in memory-mapped files.
    max_limit : int
        Maximum number of samples to return from the dataset. If zero or negative, no limit is applied.
    config : dict
        Configuration dictionary containing settings such as scaling factors.

    Attributes
    ----------
    chunks_memmaps : list of np.memmap
        Memory-mapped arrays for chunk data.
    targets_memmaps : list of np.memmap
        Memory-mapped arrays for target data.
    c_lengths_memmaps : list of np.memmap
        Memory-mapped arrays for chunk lengths data.
    t_lengths_memmaps : list of np.memmap
        Memory-mapped arrays for target lengths data.
    stdevs_memmaps : list of np.memmap
        Memory-mapped arrays for standard deviations data.
    start_indices : list of int
        Start indices for each memory-mapped array to facilitate efficient indexing.
    max_limit : int
        Maximum number of samples to return from the dataset.
    config : dict
        Configuration dictionary used for data scaling.
    data_count : int
        Total number of samples in the dataset.
    data_count_targets : int
        Total number of target samples in the dataset.

    Methods
    -------
    __getitem__(index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Retrieves the sample at the specified index.
    __len__() -> int
        Returns the number of samples in the dataset, subject to the `max_limit`.
    """

    def __init__(
        self,
        chunks_path: List[str],
        targets_path: List[str],
        c_lengths_path: List[str],
        t_lengths_path: List[str],
        stdevs_path: List[str],
        max_limit: int,
        config: Dict,
    ):  # max_length
        self.chunks_memmaps = [np.load(path, mmap_mode="r") for path in chunks_path]
        self.targets_memmaps = [np.load(path, mmap_mode="r") for path in targets_path]
        self.c_lengths_memmaps = [
            np.load(path, mmap_mode="r") for path in c_lengths_path
        ]
        self.t_lengths_memmaps = [
            np.load(path, mmap_mode="r") for path in t_lengths_path
        ]
        self.stdevs_memmaps = [np.load(path, mmap_mode="r") for path in stdevs_path]
        self.start_indices = [0] * len(chunks_path)
        # self.start_indices_targets = [0] * len(targets_path)
        self.max_limit = max_limit
        self.config = config
        self.data_count = 0
        self.data_count_targets = 0
        for index, memmap in enumerate(self.chunks_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]

        chunks = self.chunks_memmaps[memmap_index][index_in_memmap]
        targets = self.targets_memmaps[memmap_index][index_in_memmap]
        c_lengths = self.c_lengths_memmaps[memmap_index][index_in_memmap]
        t_lengths = self.t_lengths_memmaps[memmap_index][index_in_memmap]
        stdevs = self.stdevs_memmaps[memmap_index][index_in_memmap]

        targets = targets[:, np.newaxis]
        targets = targets / self.config["scaling_max_value"]  # scale target values

        stdevs = stdevs / self.config["scaling_max_value"]

        return (
            chunks.astype(np.float16),
            targets.astype(np.float32),
            c_lengths.astype(np.int16),
            t_lengths.astype(np.int16),
            stdevs.astype(np.float32),
        )

    def __len__(self):
        if self.max_limit > 0 and self.max_limit < self.data_count:
            return self.max_limit
        return self.data_count


class DataParallelIterableDataSet(IterableDataset):
    """
    A PyTorch `IterableDataset` that wraps an iterable to provide data for prediction.
    Multi-Threading not implemented yet.

    Parameters
    ----------
    iterable : iterable
        An iterable object that yields data samples.
    length : int
        The total length of the dataset.

    Attributes
    ----------
    iterable : iterable
        The iterable object used to provide data samples.
    length : int
        The length of the dataset, representing the number of samples.

    Methods
    -------
    __iter__()
        Returns the iterable object itself.
    __len__()
        Returns the length of the dataset.
    """
    def __init__(self, iterable, length, rank, world_size):
        self.iterable = iterable
        self.length = length
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # devices split
        device_rank, num_devices = (tdi.get_rank(), tdi.get_world_size()) if tdi.is_initialized() else (0, 1)  
        # workers split
        worker_info = get_worker_info()
        worker_rank, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)

        # total (devices + workers) split by device, then by worker
        num_replicas = num_workers * num_devices
        replica_rank = worker_rank * num_devices + device_rank
        # by worker, then device would be:
        # rank = device_rank * num_workers + worker_rank

        for i, data in enumerate(self.iterable):
            if i % num_replicas == replica_rank:
                #print(f"Device: {device_rank}, worker {worker_rank} fetches sample {i}")
                yield data
            else:
                continue

        # return self.iterable


    def __len__(self):
        return self.length

class IterableFastaDataSet(IterableDataset):
    """
    A PyTorch `IterableDataset` that wraps an iterable to provide data for prediction.
    Multi-Threading not implemented yet.

    Parameters
    ----------
    iterable : iterable
        An iterable object that yields data samples.
    length : int
        The total length of the dataset.

    Attributes
    ----------
    iterable : iterable
        The iterable object used to provide data samples.
    length : int
        The length of the dataset, representing the number of samples.

    Methods
    -------
    __iter__()
        Returns the iterable object itself.
    __len__()
        Returns the length of the dataset.
    """

    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return self.iterable


    def __len__(self):
        return self.length


def process_read(
    read_config: Tuple[str, str, Dict]
) -> Generator[Tuple[str, str], None, None]:
    """
    Processes a single read by splitting its sequence and yielding the results.

    Parameters
    ----------
    read_config : tuple
        A tuple containing:
        - read_seq : str
            The sequence of the read.
        - read_name : str
            The name of the read.
        - config : dict
            Configuration dictionary used for processing.

    Yields
    ------
    tuple of (str, str)
        Yields tuples containing:
        - read_name : str
            The name of the read.
        - breakpoint : str
            A segment of the split sequence.

    Notes
    -----
    - The function uses the `split_sequence` function to divide the read sequence into breakpoints.
    - If the sequence is not split (i.e., no breakpoints), a debug message is logged indicating the read was skipped.
    """
    read_seq, read_names, config = read_config

    breakpoints = split_sequence(read_seq, config)

    if breakpoints.size > 0:
        read_names = [read_names] * len(breakpoints)
        for read_name, breakpoint in zip(read_names, breakpoints):
            yield read_name, breakpoint
    else:
        logger.debug(f"Skipped read {read_names}.")


def load_fasta(
    fasta: List[Tuple[str, str]], config: Dict, total_l: int, rank:int, world_size:int
) -> Dict[str, "DataLoader"]:
    """
    Loads and processes FASTA files into a dataset for prediction, using parallel processing.

    Parameters
    ----------
    fasta : list of tuples
        A list where each tuple contains:
        - read_seq : str
            The sequence of the read.
        - read_name : str
            The name of the read.
    config : dict
        Configuration dictionary used for processing reads.
    total_l : int
        The total length of the data to be processed.

    Returns
    -------
    dict
        A dictionary containing the DataLoader configuration:
        - "dataset" : IterableFastaDataSet
            The dataset constructed from the processed FASTA reads.
        - "shuffle" : bool
            Whether to shuffle the data (set to False).
    """
    logger.debug("Splitting the reads to chunks.")

    # Define the number of worker processes
    num_processes = (
        multiprocessing.cpu_count()
    )  # Adjust this based on your system's capabilities

    read_configs = [(read_seq, read_name, config) for read_seq, read_name in fasta]

    # Process reads in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_read, read_configs)

    # Combine generators
    combined_generator = itertools.chain(*results)

    logger.debug("Splitting the reads to chunks finished.")
    
    predict_loader_kwargs = {
        #"dataset": DataParallelIterableDataSet(combined_generator, total_l, rank, world_size),
        "dataset": IterableFastaDataSet(combined_generator, total_l),
        "shuffle": False,
    }
    
    return predict_loader_kwargs


def load_numpy(
    train_limit: int,
    valid_limit: int,
    npy_train: str,
    npy_valid: Optional[str] = None,
    valid_chunks: Optional[int] = None,
    train_valid_split: float = 0.9,
    config: Optional[Dict] = None,
) -> Tuple[Dict[str, "DataLoader"], Dict[str, "DataLoader"]]:
    """
    Loads training and validation data and returns DataLoader objects for each.

    Parameters
    ----------
    train_limit : int
        Maximum number of chunks to include in the training dataset.
    valid_limit : int
        Maximum number of chunks to include in the validation dataset.
    npy_train : str
        Path to the directory containing training files.
    npy_valid : Optional[str]
        Path to the directory containing validation files. If None, the data will be split from the training set.
    valid_chunks : Optional[int]
        Number of chunks to use for validation if `npy_valid` is provided.
    train_valid_split : float
        Fraction of the training data to use for training. The rest will be used for validation when `npy_valid` is None.
    config : Optional[Dict]
        Configuration dictionary containing a random seed for splitting.

    Returns
    -------
    Tuple[Dict[str, 'DataLoader'], Dict[str, 'DataLoader']]
        A tuple containing two dictionaries:
        - Training DataLoader configuration with dataset and shuffle setting.
        - Validation DataLoader configuration with dataset and shuffle setting.
    """
    def load_paths(directory: str, prefix: str) -> List[str]:
        return sorted(
            os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)
        )
    

    chunks_train = load_paths(npy_train, "chunks-")
    targets_train = load_paths(npy_train, "targets-")
    c_lengths_train = load_paths(npy_train, "chunks_lengths-")
    t_lengths_train = load_paths(npy_train, "targets_lengths-")
    stdevs_train = load_paths(npy_train, "stdevs-")

    if npy_valid and os.path.exists(npy_valid):
        chunks_valid = load_paths(npy_valid, "chunks-")
        targets_valid = load_paths(npy_valid, "targets-")
        c_lengths_valid = load_paths(npy_valid, "chunks_lengths-")
        t_lengths_valid = load_paths(npy_valid, "targets_lengths-")
        stdevs_valid = load_paths(npy_valid, "stdevs-")
    else:
        # Lazy split for testing
        chunks_train, chunks_valid = train_test_split(
            chunks_train,
            train_size=train_valid_split,
            random_state=config["random_seed"],
        )
        targets_train, targets_valid = train_test_split(
            targets_train,
            train_size=train_valid_split,
            random_state=config["random_seed"],
        )
        c_lengths_train, c_lengths_valid = train_test_split(
            c_lengths_train,
            train_size=train_valid_split,
            random_state=config["random_seed"],
        )
        t_lengths_train, t_lengths_valid = train_test_split(
            t_lengths_train,
            train_size=train_valid_split,
            random_state=config["random_seed"],
        )
        stdevs_train, stdevs_valid = train_test_split(
            stdevs_train,
            train_size=train_valid_split,
            random_state=config["random_seed"],
        )

    chunks_train, targets_train, c_lengths_train, t_lengths_train, stdevs_train = (
        sort_files(
            chunks_train, targets_train, c_lengths_train, t_lengths_train, stdevs_train
        )
    )
    check_file_order(chunks_train, targets_train)
    chunks_valid, targets_valid, c_lengths_valid, t_lengths_valid, stdevs_valid = (
        sort_files(
            chunks_valid, targets_valid, c_lengths_valid, t_lengths_valid, stdevs_valid
        )
    )
    check_file_order(chunks_valid, targets_valid)
    train_loader_kwargs = {
        "dataset": ChunkDataSetMemmap(
            chunks_train,
            targets_train,
            c_lengths_train,
            t_lengths_train,
            stdevs_train,
            train_limit,
            config,
        ),
        "shuffle": True,
    }
    valid_loader_kwargs = {
        "dataset": ChunkDataSetMemmap(
            chunks_valid,
            targets_valid,
            c_lengths_valid,
            t_lengths_valid,
            stdevs_valid,
            valid_limit,
            config,
        ),
        "shuffle": False,
    }
    return train_loader_kwargs, valid_loader_kwargs


def extract_number(filename):
    return int(filename.split("-")[-1].split(".")[0])


def check_file_order(chunks_path: list[str], targets_path: list[str]) -> None:
    """
    Checks the consistency of file ordering between chunk files and target files.

    Parameters
    ----------
    chunks_path : list of str
        List of paths to chunk files.
    targets_path : list of str
        List of paths to target files.

    Returns
    -------
    None
    """
    inconsistent_position = None
    for i, (chunk_file, target_file) in enumerate(zip(chunks_path, targets_path)):
        chunk_number = extract_number(chunk_file)
        target_number = extract_number(target_file)
        if chunk_number != target_number:
            inconsistent_position = i
            break
    if inconsistent_position is not None:
        logger.warning(
            f"The order becomes inconsistent at position {inconsistent_position}."
        )
    else:
        logger.debug(f"The order of {chunks_path} is consistent.")


def sort_files(
    chunks_path: list[str],
    targets_path: list[str],
    c_lengths_path: list[str],
    t_lengths_path: list[str],
    stdevs_path: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Sorts file paths based on their filenames and returns the sorted lists.

    Parameters
    ----------
    chunks_path : list of str
        List of paths to chunk files.
    targets_path : list of str
        List of paths to target files.
    c_lengths_path : list of str
        List of paths to chunk lengths files.
    t_lengths_path : list of str
        List of paths to target lengths files.
    stdevs_path : list of str
        List of paths to standard deviations files.

    Returns
    -------
    tuple of lists of str
        A tuple containing the sorted lists of file paths:
        - Sorted chunk file paths.
        - Sorted target file paths.
        - Sorted chunk lengths file paths.
        - Sorted target lengths file paths.
        - Sorted standard deviations file paths.
    """
    def sort_by_filename(paths):
        return sorted(paths, key=lambda path: path.split("/")[-1])

    return tuple(
        sort_by_filename(path_list) for path_list in
        [chunks_path, targets_path, c_lengths_path, t_lengths_path, stdevs_path]
    )
