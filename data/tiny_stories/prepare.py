"""
TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
Downloads and tokenizes the data and saves data shards to disk.
$ python -m data.tiny_stories.prepare
"""

import os

from datasets import load_dataset

from data.tokenizer.tiktoken import shard_dataset

if __name__ == "__main__":
    shard_size = int(1e8)  # 100M tokens per shard, total of 5 shards
    fw = load_dataset("roneneldan/TinyStories", split="train")
    cur_dir = os.path.dirname(__file__)
    os.makedirs(cur_dir, exist_ok=True)
    shard_dataset(fw, split="train", shard_size=shard_size, out_dir=cur_dir)
