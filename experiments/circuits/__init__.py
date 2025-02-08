from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch

from circuits.features.cache import ModelCache
from models.sparsified import SparsifiedGPT


@dataclass(frozen=True)
class Feature:
    """
    Represents a feature at a specific location.
    """

    layer_idx: int
    token_idx: int
    feature_idx: int

    def as_tuple(self) -> tuple[int, int]:
        return self.token_idx, self.feature_idx

    def __repr__(self) -> str:
        return f"({f'{self.token_idx},': <4}{self.feature_idx: >4})"


@torch.no_grad()
def calculate_kl_divergence(
    model: SparsifiedGPT,
    model_cache: ModelCache,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_features: Sequence[Feature] = (),
) -> tuple[float, dict[str, float]]:
    """
    Calculate KL divergence between target logits and logits produced by model using reconstructed activations.
    """
    # Copy feature magnitudes to avoid modifying original
    feature_magnitudes = feature_magnitudes.clone()

    # Patch using resampling technique
    patched_feature_magnitudes = patch_via_resampling(
        model_cache, feature_magnitudes, circuit_features, num_samples=8
    )  # Shape: (B, T, F)

    # Reconstruct activations and compute logits
    x_reconstructed = model.saes[str(layer_idx)].decode(patched_feature_magnitudes)  # type: ignore
    predicted_logits = model.gpt.forward_with_patched_activations(
        x_reconstructed, layer_idx=layer_idx
    )  # Shape: (B, T, V)

    # We only care about logits for the target token
    predicted_logits = predicted_logits[:, target_token_idx, :]  # Shape: (B, V)

    # Convert logits to probabilities before averaging across samples
    predicted_probabilities = torch.nn.functional.softmax(predicted_logits, dim=-1)
    predicted_probabilities = predicted_probabilities.mean(dim=0)  # Shape: (V)
    predicted_logits = torch.log(predicted_probabilities)  # Shape: (V)

    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(predicted_logits, dim=-1),
        torch.nn.functional.softmax(target_logits, dim=-1),
        reduction="sum",
    )

    # Calculate predictions
    predictions = get_predictions(model, predicted_logits)

    return kl_div.item(), predictions


def get_predictions(
    model: SparsifiedGPT,
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
        results[model.gpt.config.tokenizer.decode_token(int(i.item()))] = round(p.item() * 100, 2)
    return results


def estimate_ablation_effects(
    model: SparsifiedGPT,
    model_cache: ModelCache,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,
    feature_magnitudes: torch.Tensor,
    circuit_features: list[Feature],
) -> dict[Feature, float]:
    """
    Map features to KL divergence.
    """
    feature_to_kl_div: dict[Feature, float] = {}
    for feature in circuit_features:
        patched_features = [f for f in circuit_features if f != feature]
        kl_div, _ = calculate_kl_divergence(
            model,
            model_cache,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            circuit_features=patched_features,
        )
        feature_to_kl_div[feature] = kl_div
    return feature_to_kl_div


def patch_via_resampling(
    model_cache: ModelCache,
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_features: Sequence[Feature],
    num_samples: int = 8,
) -> torch.Tensor:  # Shape: (B, T, F)
    """
    Resample feature magnitudes using cached values, returning N samples.
    Samples are drawn from the top 100 most similar rows in the cache.

    Based on technique described here: https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN
    """
    # Group circuit features by token index
    circuit_feature_groups: dict[int, set[Feature]] = defaultdict(set)
    for feature in circuit_features:
        circuit_feature_groups[feature.token_idx].add(feature)

    for token_idx, circuit_feature_group in circuit_feature_groups.items():
        pass

    # TODO: Implement correctly
    return patch_via_zero_ablation(feature_magnitudes, circuit_features, num_samples)


def patch_via_zero_ablation(
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_features: Sequence[Feature],
    num_samples: int = 8,
) -> torch.Tensor:  # Shape: (B, T, F)
    """
    Set non-circuit features to zero and duplicate the result S times. (Useful for debugging)
    """
    # Zero-ablate non-circuit features
    patched_feature_magnitudes = torch.zeros_like(feature_magnitudes)
    if circuit_features:
        token_idxs = [f.token_idx for f in circuit_features]
        feature_idxs = [f.feature_idx for f in circuit_features]
        patched_feature_magnitudes[token_idxs, feature_idxs] = feature_magnitudes[token_idxs, feature_idxs]

    # Duplicate feature magnitudes `num_samples` times
    patched_samples = patched_feature_magnitudes.unsqueeze(0).expand(num_samples, -1, -1)  # Shape: (B, T, F)
    return patched_samples
