"""
Find all features needed to reconstruct the output logits of a model to within a certain KL divergence threshold.

$ python -m experiments.circuits.search --sequence_idx=0 --token_idx=51 --layer_idx=0 --threshold=0.1
"""

import argparse

import torch

from circuits.features import Feature
from circuits.features.ablation import ResampleAblator
from circuits.features.cache import ModelCache
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from experiments.circuits import (
    calculate_kl_divergences,
    estimate_ablation_effects,
    get_predictions,
)
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_idx", type=int, help="Index for start of sequence [0...shard.tokens.size)")
    parser.add_argument("--token_idx", type=int, help="Index for token in the sequence [0...block_size)")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    parser.add_argument("--layer_idx", type=int, default=0, help="SAE layer to analyze")
    parser.add_argument("--threshold", type=float, default=0.1, help="Max threshold for KL divergence")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    threshold = args.threshold
    layer_idx = args.layer_idx
    target_token_idx = args.token_idx

    # Load model
    defaults = Config()
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    model: SparsifiedGPT = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Load cached feature magnitudes
    model_cache = ModelCache(checkpoint_dir)

    # Set feature ablation strategy
    ablator = ResampleAblator(model_cache, k_nearest=128)

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore

    # Load tokens
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    tokens = shard.tokens[args.sequence_idx : args.sequence_idx + model.config.block_size].to(defaults.device)
    decoded_tokens = model.gpt.config.tokenizer.decode_sequence(tokens.tolist())
    decoded_target = model.gpt.config.tokenizer.decode_token(int(tokens[target_token_idx].item()))
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")
    print(f"Target layer: {layer_idx}")
    print(f"Target threshold: {threshold}")

    # Get target logits
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(tokens.unsqueeze(0))
    target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
    target_predictions = get_predictions(model, target_logits)
    print(f"Target predictions: {target_predictions}\n")

    # Get output for layer
    feature_magnitudes = output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)

    # Get non-zero features that are before or at the target token
    non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
    all_features: list[Feature] = [
        Feature(layer_idx, t.item(), f.item()) for t, f in zip(*non_zero_indices) if t <= target_token_idx
    ]

    # Starting search states
    # NOTE: Configured to remove one feature at a time
    search_target = len(all_features)  # Start with all features
    search_interval_start: float = 1.0
    search_interval_end: float = 1.0
    search_max_steps = len(all_features)
    circuit_kl_div: float = float("inf")
    circuit_features: list[Feature] = all_features[:]
    discarded_features: list[Feature] = []
    search_interval: float = search_interval_start
    search_step = 0

    # Start search
    while search_step < search_max_steps:
        # Compute KL divergence
        circuit_variant = frozenset(circuit_features[:search_target])
        kld_result = calculate_kl_divergences(
            model,
            ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            [circuit_variant],
        )[circuit_variant]
        circuit_kl_div = kld_result.kl_divergence

        # Print results
        print(
            f"Search: {search_target}/{len(all_features)} ({search_interval:.2f}) - "
            f"Circuit KL div: {round(circuit_kl_div, 4)} - "
            f"Predictions: {kld_result.predictions}"
        )

        # Update search index
        if circuit_kl_div < threshold:
            # Update candidate features
            discardable_features = circuit_features[search_target:]
            discarded_features += discardable_features
            circuit_features = circuit_features[:search_target]

            # Sort features by KL divergence (descending)
            estimated_ablation_effects = estimate_ablation_effects(
                model,
                ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                circuit_features=circuit_features,
            )
            circuit_features.sort(key=lambda x: estimated_ablation_effects[x], reverse=True)

            # Include fewer features
            search_target -= round(search_interval)
        else:
            # Include more features
            search_target += round(search_interval)

        # Clamp search index
        search_target = min(max(search_target, 0), len(circuit_features))

        # Update search interval
        search_step += 1
        search_interval = search_interval_start + (search_interval_end - search_interval_start) * (
            search_step / (search_max_steps - 1)
        )

        # Check for early stopping
        if circuit_kl_div < threshold and min(estimated_ablation_effects.values()) > threshold:
            print("Stopping early. Can't improve KL divergence.")
            break
        if circuit_kl_div > threshold and len(circuit_features) == len(all_features):
            print("Stopping early. Baseline KL divergence is too high.")
            break

    # Check that all features are accounted for
    assert len(discarded_features) + len(circuit_features) == len(all_features)

    # Print final results (grouped by token_idx)
    print(f"\nCircuit features ({len(circuit_features)}):")
    for token_idx in range(max([f.token_idx for f in circuit_features]) + 1):
        features = [f for f in circuit_features if f.token_idx == token_idx]
        if len(features) > 0:
            print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in features])}")
