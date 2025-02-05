"""
Find all features needed to reconstruct the output logits of a model to within a certain KL divergence threshold using
binary search.

$ python -m experiments.circuits.basic --sequence_idx=0 --token_idx=51 --layer_idx=0
"""

import argparse

import torch

from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from experiments.circuits import MaskedFeature, calculate_kl_divergence, get_predictions
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

    # Load model
    defaults = Config()
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    model: SparsifiedGPT = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore

    # Load tokens
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    layer_idx = args.layer_idx
    target_token_idx = args.token_idx
    tokens = shard.tokens[args.sequence_idx : args.sequence_idx + model.config.block_size].to(defaults.device)
    decoded_tokens = model.gpt.config.tokenizer.decode_sequence(tokens.tolist())
    decoded_target = model.gpt.config.tokenizer.decode_token(int(tokens[target_token_idx].item()))
    print(f"Target layer: {layer_idx}")
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")

    # Get target logits
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(tokens.unsqueeze(0))
    target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
    target_predictions = get_predictions(model, target_logits)
    print(f"Target predictions: {target_predictions}")

    # Get output for layer
    feature_magnitudes = output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)

    # Compute KL divergence if using all features
    baseline_kl_div, predictions = calculate_kl_divergence(
        model,
        layer_idx,
        target_token_idx,
        target_logits,
        feature_magnitudes,
    )

    # Get non-zero features that are before or at the target token
    non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
    all_features = [(t.item(), f.item()) for t, f in zip(*non_zero_indices) if t <= target_token_idx]

    # Meature ablation effects
    feature_to_kl_div: dict[tuple[int, int], float] = {}
    for token_idx, feature_idx in all_features:
        masked_features = [MaskedFeature(token_idx=token_idx, feature_idx=feature_idx)]
        kl_div, _ = calculate_kl_divergence(
            model,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            masked_features,
        )
        feature_to_kl_div[(token_idx, feature_idx)] = kl_div

    # Sort features by KL divergence (descending)
    all_features.sort(key=lambda x: feature_to_kl_div[x], reverse=True)

    # Starting search states
    max_idx = len(all_features) - 1
    search_idx = max_idx
    search_interval = (max_idx + 1) // 2
    circuit_kl_div = baseline_kl_div
    masked_features: list[MaskedFeature] = []

    print(f"Target threshold: {args.threshold}")
    print(f"Baseline KL div: {round(baseline_kl_div, 4)} - {predictions}")

    # Start binary search
    while search_interval > 0:
        # TODO: Resort features by KL divergence every loop

        # Update search index
        if circuit_kl_div < args.threshold:
            # Include fewer features
            search_idx -= search_interval
        else:
            # Include more features
            search_idx += search_interval

        # Clamp search index
        search_idx = min(max(search_idx, 0), max_idx)

        # Narrow search interval
        search_interval = (search_interval + 1) // 2 if search_interval > 1 else 0

        # Get features to ablate
        masked_features = [
            MaskedFeature(token_idx=token_idx, feature_idx=feature_idx)
            for token_idx, feature_idx in all_features[search_idx + 1 :]
        ]

        # Compute KL divergence
        circuit_kl_div, predictions = calculate_kl_divergence(
            model,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            masked_features,
        )

        # Print results
        print(
            f"Search idx: {search_idx}/{max_idx} - Circuit KL div: {round(circuit_kl_div, 4)} - "
            f"Predictions: {predictions}"
        )

    # Print final results
    print("\nFinal features:")
    for token_idx, feature_idx in all_features[: search_idx + 1]:
        print(f"({token_idx}, {feature_idx}): {feature_to_kl_div[(token_idx, feature_idx)]:.4f}")

    print("\nMasked features:")
    for masked_feature in masked_features:
        print(
            f"({masked_feature.token_idx}, {masked_feature.feature_idx}): {feature_to_kl_div[masked_feature.as_tuple()]:.4f}"
        )
