"""
Export feature metrics.

$ python -m experiments.circuits.export
"""

import argparse
from pathlib import Path

from circuits.features.samples import ModelSampleSet
from config import TrainingConfig
from data.dataloaders import DatasetShard
from data.tokenizers import TokenizerType


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to inspect")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--tokenizer", type=str, default="ascii", help="Dataset split to use")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load samples
    sample_set = ModelSampleSet(checkpoint_dir=TrainingConfig.checkpoints_dir / args.model)
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)
    tokenizer = TokenizerType(args.tokenizer).as_tokenizer()
    sample_set.export(Path("exports") / args.model / "samples", shard, tokenizer)
