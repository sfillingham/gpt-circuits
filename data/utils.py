"""
This module contians utilities for processing datasets.
"""

import numpy as np
from datasets import Dataset


def save_dataset(dataset: Dataset, out_dir: str, file_prefix: str, num_shards: int, key: str = "ids") -> None:
    """
    Shards a tokenized dataset and saves concatenated sequences to disk as a single numpy array.
    """
    for i in range(num_shards):
        shard = dataset.shard(num_shards, i)
        num_tokens = sum(len(ids) for ids in shard[key])
        ids = np.zeros(num_tokens, dtype=np.uint16)
        cursor = 0
        for seq in shard[key]:
            ids[cursor : cursor + len(seq)] = seq
            cursor += len(seq)
        filename = f"{file_prefix}_{i:06d}.npy"
        np.save(f"{out_dir}/{filename}", ids)
        print(f"Saved {len(ids)} tokens to {filename}")
