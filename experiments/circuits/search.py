"""
Find all features needed to reconstruct the output logits of a model to within a certain KL divergence threshold.

$ python -m experiments.circuits.search --sequence_idx=0 --token_idx=51 --layer_idx=0
"""

import argparse

import torch

from circuits.features import Feature
from circuits.features.ablation import ResampleAblator, ZeroAblator  # noqa: F401
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from experiments.circuits import (
    analyze_circuits,
    estimate_feature_ablation_effects,
    estimate_token_ablation_effects,
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
    parser.add_argument("--threshold", type=float, default=0.15, help="Max threshold for KL divergence")
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

    # Load cached metrics and feature magnitudes
    model_profile = ModelProfile(checkpoint_dir)
    model_cache = ModelCache(checkpoint_dir)

    # Set feature ablation strategy
    # ablator = ZeroAblator()
    k_nearest = 128  # How many nearest neighbors to consider in resampling
    num_samples = 128  # Number of samples to use for estimating KL divergence
    positional_coefficient = 2.0  # How important is the position of a feature
    ablator = ResampleAblator(
        model_profile,
        model_cache,
        k_nearest=k_nearest,
        positional_coefficient=positional_coefficient,
    )

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore

    # Load tokens
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    tokenizer = model.gpt.config.tokenizer
    tokens = shard.tokens[args.sequence_idx : args.sequence_idx + model.config.block_size].to(defaults.device)
    decoded_tokens = tokenizer.decode_sequence(tokens.tolist())
    decoded_target = tokenizer.decode_token(int(tokens[target_token_idx].item()))
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")
    print(f"Target layer: {layer_idx}")
    print(f"Target threshold: {threshold}")

    # Get target logits
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(tokens.unsqueeze(0))
    target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
    target_predictions = get_predictions(tokenizer, target_logits)
    print(f"Target predictions: {target_predictions}")

    # Get baseline KL divergence
    x_reconstructed = model.saes[str(layer_idx)].decode(output.feature_magnitudes[layer_idx])  # type: ignore
    predicted_logits = model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx)
    predicted_logits = predicted_logits[0, target_token_idx, :]  # Shape: (V)
    baseline_predictions = get_predictions(tokenizer, predicted_logits)
    baseline_kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(target_logits, dim=-1),
        torch.nn.functional.softmax(predicted_logits, dim=-1),
        reduction="sum",
    )
    print(f"Baseline predictions: {baseline_predictions}")
    print(f"Baseline KL divergence: {baseline_kl_div.item():.4f}\n")

    # Get output for layer
    feature_magnitudes = output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)

    # Get non-zero features that are before or at the target token
    non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
    all_features: set[Feature] = {
        Feature(layer_idx, t.item(), f.item()) for t, f in zip(*non_zero_indices) if t <= target_token_idx
    }

    # Circuit to start pruning
    circuit_features: set[Feature] = all_features

    ### Part 1: Start by searching for important tokens
    print("Starting search for important tokens...")

    # Group features by token index
    features_by_token_idx: dict[int, set[Feature]] = {}
    for token_idx in range(target_token_idx + 1):
        features_by_token_idx[token_idx] = set({f for f in all_features if f.token_idx == token_idx})

    # Starting search states
    discard_candidates: set[Feature] = set({})
    circuit_kl_div: float = float("inf")

    # Start search
    for search_step in range(target_token_idx + 1):
        # Compute KL divergence
        circuit_candidate = frozenset(circuit_features - discard_candidates)
        circuit_analysis = analyze_circuits(
            model,
            ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            [circuit_candidate],
            num_samples=num_samples,
        )[circuit_candidate]
        circuit_kl_div = circuit_analysis.kl_divergence
        num_unique_tokens = len(set(f.token_idx for f in circuit_candidate))

        # Print results
        print(
            f"Tokens: {num_unique_tokens}/{target_token_idx + 1} - "
            f"KL Div: {circuit_kl_div:.4f} - "
            f"Predictions: {circuit_analysis.predictions}"
        )

        # If below threshold, continue search
        if circuit_kl_div < threshold:
            # Update candidate features
            circuit_features = set(circuit_candidate)

            # Sort features by KL divergence (descending)
            estimated_token_ablation_effects = estimate_token_ablation_effects(
                model,
                ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                circuit_features=circuit_features,
                num_samples=num_samples,
            )
            least_important_token_idx = min(estimated_token_ablation_effects.items(), key=lambda x: x[1])[0]
            least_important_token_kl_div = estimated_token_ablation_effects[least_important_token_idx]
            discard_candidates = features_by_token_idx[least_important_token_idx]

            # Check for early stopping
            if least_important_token_kl_div > threshold:
                print("Stopping early. Can't improve KL divergence.")
                break

        # If above threshold, stop search
        else:
            print("Stopping early. KL divergence is too high.")
            break

    # Print results (grouped by token_idx)
    print(f"\nCircuit after token search ({len(circuit_features)}):")
    for token_idx in range(max([f.token_idx for f in circuit_features]) + 1):
        features = [f for f in circuit_features if f.token_idx == token_idx]
        if len(features) > 0:
            print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in features])}")
    print("")

    ### Part 2: Search for important features
    print("Starting search for important features...")

    # Starting search states
    discard_candidates: set[Feature] = set({})
    circuit_kl_div: float = float("inf")

    # # Start search
    for search_step in range(len(circuit_features)):
        # Compute KL divergence
        circuit_candidate = frozenset(circuit_features - discard_candidates)
        circuit_analysis = analyze_circuits(
            model,
            ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            [circuit_candidate],
            num_samples=num_samples,
        )[circuit_candidate]
        circuit_kl_div = circuit_analysis.kl_divergence

        # Print results
        print(
            f"Features: {len(circuit_candidate)}/{len(all_features)} - "
            f"KL Div: {circuit_kl_div:.4f} - "
            f"Predictions: {circuit_analysis.predictions}"
        )

        # If below threshold, continue search
        if circuit_kl_div < threshold:
            # Update candidate features
            circuit_features = set(circuit_candidate)

            # Sort features by KL divergence (descending)
            estimated_feature_ablation_effects = estimate_feature_ablation_effects(
                model,
                ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                circuit_features=circuit_features,
                num_samples=num_samples,
            )
            least_important_feature = min(estimated_feature_ablation_effects.items(), key=lambda x: x[1])[0]
            least_important_feature_kl_div = estimated_feature_ablation_effects[least_important_feature]
            discard_candidates = {least_important_feature}

            # Check for early stopping
            if least_important_feature_kl_div > threshold:
                print("Stopping early. Can't improve KL divergence.")
                break

        # If above threshold, stop search
        else:
            print("Stopping early. KL divergence is too high.")
            break

    # Print final results (grouped by token_idx)
    print(f"\nCircuit after feature search ({len(circuit_features)}):")
    for token_idx in range(max([f.token_idx for f in circuit_features]) + 1):
        features = [f for f in circuit_features if f.token_idx == token_idx]
        if len(features) > 0:
            print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in features])}")
