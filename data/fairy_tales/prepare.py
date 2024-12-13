"""
Prepare the Fairy Tales dataset.

$ python -m data.fairy_tales.prepare
"""

import os
import re

import numpy as np
import tiktoken

# Fairy tales dataset in repository
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Remove all newlines (which are used for line breaks) while keeping double newlines
data = re.sub(r"(?<!\n)\n(?!\n)", " ", data)

# Use tiktoken to tokenize the data
encoding = tiktoken.get_encoding("gpt2")


def encode(s):
    return encoding.encode(s, allowed_special={"<|endoftext|>"})


# create the train and test splits
n = len(data)
dataset_len = n
train_len = int(n * 0.9)
trimmed_train_len = train_len - (train_len % 256)  # round down to nearest multiple of 256
train_data = data[:trimmed_train_len]
val_data = data[trimmed_train_len:]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
out_dir = os.path.dirname(__file__)
np.save(os.path.join(out_dir, f"train_{0:06d}"), train_ids)
np.save(os.path.join(out_dir, f"val_{0:06d}"), val_ids)
