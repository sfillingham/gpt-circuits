"""
Downloads, tokenizes, and saves pile-10k dataset to disk.
https://huggingface.co/datasets/NeelNanda/pile-10k

$ python -m data.pile_10k.prepare
"""

import os

from datasets import Dataset, load_dataset

from data.tokenizers import TikTokenTokenizer
from data.utils import save_dataset

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    # Only train split is available
    dataset: Dataset = load_dataset("NeelNanda/pile-10k", split="train")  # type: ignore

    # Use about half the available CPUs for tokenization
    num_proc = max(1, (os.cpu_count() or 2) // 2)

    # Use the TikToken tokenizer
    tokenizer = TikTokenTokenizer()

    # Tokenization function
    def tokenize(example):
        # Add an end-of-text token after every sample.
        ids = tokenizer.encode(example["text"])
        ids.append(tokenizer.encoding.eot_token)
        return {"ids": ids}

    # Tokenize and split the dataset
    tokenized_datasets = dataset.map(tokenize, num_proc=num_proc).train_test_split(test_size=0.1, seed=42)

    # Save the training dataset
    save_dataset(tokenized_datasets["train"], out_dir, "train", num_shards=1)

    # Save the validation dataset
    save_dataset(tokenized_datasets["test"], out_dir, "val", num_shards=1)
