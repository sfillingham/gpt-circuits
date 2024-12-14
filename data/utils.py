"""
This module contians utilities for processing datasets.
"""

import itertools

import numpy as np
from datasets import Dataset


def save_dataset(dataset: Dataset, out_dir: str, file_prefix: str, num_shards: int, key: str = "ids") -> None:
    """
    Shards a tokenized dataset and saves concatenated sequences to disk as a single numpy array.
    """
    for i in range(num_shards):
        shard = dataset.shard(num_shards, i)
        concatenated_ids = list(itertools.chain.from_iterable(shard[key]))
        ids = np.array(concatenated_ids, dtype=np.uint16)
        filename = f"{file_prefix}_{i:06d}.npy"
        np.save(f"{out_dir}/{filename}", ids)
        print(f"Saved {len(ids)} tokens to {filename}")
