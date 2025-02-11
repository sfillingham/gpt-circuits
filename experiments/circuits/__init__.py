from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm import tqdm

from circuits.features import Feature
from circuits.features.ablation import Ablator
from data.tokenizers import Tokenizer
from models.sparsified import SparsifiedGPT


@dataclass(frozen=True)
class CircuitResult:
    kl_divergence: float
    predictions: dict[str, float]


@torch.no_grad()
def analyze_circuits(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_variants: Sequence[frozenset[Feature]],
) -> dict[frozenset[Feature], CircuitResult]:
    """
    Calculate KL divergence between target logits and logits produced by model using reconstructed activations.
    """
    # For storing results
    results: dict[frozenset[Feature], CircuitResult] = {}

    # Patch feature magnitudes for each circuit variant
    patched_feature_magnitudes = patch_feature_magnitudes(
        ablator,
        layer_idx,
        target_token_idx,
        feature_magnitudes,
        circuit_variants,
        num_samples=32,
    )

    # Get predicted logits for each circuit variant when using patched feature magnitudes
    predicted_logits = get_predicted_logits(
        model,
        layer_idx,
        patched_feature_magnitudes,
        target_token_idx,
    )

    # Calculate KL divergence and predictions for each variant
    for circuit_variant, circuit_logits in predicted_logits.items():
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(circuit_logits, dim=-1),
            torch.nn.functional.softmax(target_logits, dim=-1),
            reduction="sum",
        )

        # Calculate predictions
        predictions = get_predictions(model.gpt.config.tokenizer, circuit_logits)

        # Store results
        results[circuit_variant] = CircuitResult(kl_divergence=kl_div.item(), predictions=predictions)

    return results


def patch_feature_magnitudes(
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    feature_magnitudes: torch.Tensor,
    circuit_variants: Sequence[frozenset[Feature]],
    num_samples: int,
) -> dict[frozenset[Feature], torch.Tensor]:  # Shape: (num_samples, T, F)
    """
    Patch feature magnitudes for a list of circuit variants.
    """
    # For mapping variants to patched feature magnitudes
    patched_feature_magnitudes: dict[frozenset[Feature], torch.Tensor] = {}

    # Patch feature magnitudes for each variant
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                ablator.patch,
                layer_idx=layer_idx,
                target_token_idx=target_token_idx,
                feature_magnitudes=feature_magnitudes,
                circuit_features=circuit_variant,
                num_samples=num_samples,
            ): circuit_variant
            for circuit_variant in circuit_variants
        }

        # Dont' show progress bar if only one circuit variant is provided
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Patching feature magnitudes",
            disable=len(circuit_variants) <= 1,
        ):
            circuit_variant = futures[future]
            patched_feature_magnitudes[circuit_variant] = future.result()

    # Return patched feature magnitudes
    return patched_feature_magnitudes


def get_predicted_logits(
    model: SparsifiedGPT,
    layer_idx: int,
    patched_feature_magnitudes: dict[frozenset[Feature], torch.Tensor],  # Shape: (num_samples, T, F)
    target_token_idx: int,
) -> dict[frozenset[Feature], torch.Tensor]:  # Shape: (V)
    """
    Get predicted logits for a set of circuit variants when using patched feature magnitudes.

    TODO: Use batching to improve performance
    """
    results: dict[frozenset[Feature], torch.Tensor] = {}

    for circuit_variant, circuit_feature_magnitudes in patched_feature_magnitudes.items():
        # Reconstruct activations and compute logits
        x_reconstructed = model.saes[str(layer_idx)].decode(circuit_feature_magnitudes)  # type: ignore
        predicted_logits = model.gpt.forward_with_patched_activations(
            x_reconstructed, layer_idx=layer_idx
        )  # Shape: (num_samples, T, V)

        # We only care about logits for the target token
        predicted_logits = predicted_logits[:, target_token_idx, :]  # Shape: (num_samples, V)

        # Convert logits to probabilities before averaging across samples
        predicted_probabilities = torch.nn.functional.softmax(predicted_logits, dim=-1)
        predicted_probabilities = predicted_probabilities.mean(dim=0)  # Shape: (V)
        predicted_logits = torch.log(predicted_probabilities)  # Shape: (V)

        # Store results
        results[circuit_variant] = predicted_logits

    return results


def get_predictions(
    tokenizer: Tokenizer,
    logits: torch.Tensor,  # Shape: (V)
    count: int = 5,
) -> dict[str, float]:
    """
    Map logits to probabilities and return top 5 predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=count)
    results: dict[str, float] = {}
    for i, p in zip(topk.indices, topk.values):
        results[tokenizer.decode_token(int(i.item()))] = round(p.item() * 100, 2)
    return results


def estimate_feature_ablation_effects(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,
    feature_magnitudes: torch.Tensor,
    circuit_features: set[Feature],
) -> dict[Feature, float]:
    """
    Map features to KL divergence.
    """
    # Generate all variations with one feature removed
    circuit_variants: dict[Feature, frozenset[Feature]] = {}
    for feature in circuit_features:
        circuit_variants[feature] = frozenset([f for f in circuit_features if f != feature])

    # Calculate KL divergence for each variant
    kld_results = analyze_circuits(
        model,
        ablator,
        layer_idx,
        target_token_idx,
        target_logits,
        feature_magnitudes,
        [variant for variant in circuit_variants.values()],
    )

    # Map features to KL divergence
    return {feature: kld_results[variant].kl_divergence for feature, variant in circuit_variants.items()}


def estimate_token_ablation_effects(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,
    feature_magnitudes: torch.Tensor,
    circuit_features: set[Feature],
) -> dict[int, float]:
    """
    Map features to KL divergence.
    """
    # Generate all variations with one token removed
    circuit_variants: dict[int, frozenset[Feature]] = {}
    unique_token_indices = {f.token_idx for f in circuit_features}
    for token_idx in unique_token_indices:
        circuit_variant = frozenset([f for f in circuit_features if f.token_idx != token_idx])
        circuit_variants[token_idx] = circuit_variant

    # Calculate KL divergence for each variant
    kld_results = analyze_circuits(
        model,
        ablator,
        layer_idx,
        target_token_idx,
        target_logits,
        feature_magnitudes,
        [variant for variant in circuit_variants.values()],
    )

    # Map token indices to KL divergence
    return {token_idx: kld_results[variant].kl_divergence for token_idx, variant in circuit_variants.items()}
