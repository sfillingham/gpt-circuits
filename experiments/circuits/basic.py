"""
Find all features needed to reconstruct the output logits of a model to within a certain KL divergence threshold using
a brute-force technique.

$ python -m experiments.circuits.basic --token_idx=0 --split=train --layer_idx=0
"""

import argparse

import torch

from circuits.features import FeatureSet
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_idx", type=int, help="Starting index for token sequence to analyze")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    parser.add_argument("--layer_idx", type=int, default=0, help="SAE layer to analyze")
    parser.add_argument("--threshold", type=float, default=1, help="Max threshold for KL divergence")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load model
    defaults = Config()
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    model = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)

    # Load tokens
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    tokens = shard.tokens[args.token_idx : args.token_idx + model.config.block_size].to(defaults.device)
    decoded_tokens = model.gpt.config.tokenizer.decode_sequence(tokens.tolist())
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')

    # Get target logits
    output: SparsifiedGPTOutput = model(tokens.unsqueeze(0))
    target_logits = output.logits.squeeze(0)

    # Compute KL divergence if using all features
    layer_idx = args.layer_idx
    x_reconstructed = output.reconstructed_activations[layer_idx]
    predicted_logits = model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx).squeeze(0)
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(predicted_logits, dim=-1),
        torch.nn.functional.softmax(target_logits, dim=-1),
        reduction="batchmean",
    )
    print(f"KL divergence using all features for layer {layer_idx}: {round(kl_div.item(), 4)}")

    # Load feature metrics
    feature_set = FeatureSet(checkpoint_dir)
