from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
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
    masked_features: Sequence[Feature] = (),
) -> tuple[float, dict[str, float]]:
    """
    Calculate KL divergence between target logits and logits produced by model using reconstructed activations.
    """
    # Copy feature magnitudes to avoid modifying original
    feature_magnitudes = feature_magnitudes.clone()

    # Estimate masked feature magnitudes
    estimates = estimate_masked_feature_magnitudes(model_cache, feature_magnitudes, masked_features)

    # Ablate masked features
    for masked_feature in masked_features:
        feature_magnitudes[masked_feature.token_idx, masked_feature.feature_idx] = estimates[masked_feature]

    # Reconstruct activations and compute logits
    feature_magnitudes = feature_magnitudes.unsqueeze(0)  # Shape: (1, T, F)
    x_reconstructed = model.saes[str(layer_idx)].decode(feature_magnitudes)  # type: ignore
    predicted_logits = model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx)
    predicted_logits = predicted_logits.squeeze(0)[target_token_idx]  # Shape: (V)

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
    masked_features: list[Feature],
) -> dict[Feature, float]:
    """
    Map features to KL divergence.
    """
    feature_to_kl_div: dict[Feature, float] = {}
    for feature in circuit_features:
        kl_div, _ = calculate_kl_divergence(
            model,
            model_cache,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            masked_features=masked_features + [feature],
        )
        feature_to_kl_div[feature] = kl_div
    return feature_to_kl_div


def estimate_masked_feature_magnitudes(
    model_cache: ModelCache,
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    masked_features: Sequence[Feature],
) -> dict[Feature, float]:
    """
    Estimate the magnitudes of masked features.
    """
    if not masked_features:
        return {}

    layer_idx = masked_features[0].layer_idx
    estimates: dict[Feature, float] = {f: 0.0 for f in masked_features}

    # Group masked features by token index
    masked_feature_groups: dict[int, set[Feature]] = defaultdict(set)
    for feature in masked_features:
        masked_feature_groups[feature.token_idx].add(feature)

    for token_idx, masked_feature_group in masked_feature_groups.items():
        # Get feature indices
        active_feature_idxs = feature_magnitudes[token_idx].nonzero().flatten().tolist()
        masked_feature_idxs = [f.feature_idx for f in masked_feature_group]
        unmasked_feature_idxs = [i for i in active_feature_idxs if i not in masked_feature_idxs]
        if unmasked_feature_idxs:
            target_magnitudes = feature_magnitudes[token_idx, unmasked_feature_idxs].cpu().numpy()
            cached_magnitudes = model_cache[layer_idx].csc_matrix[:, unmasked_feature_idxs]

            # Find the top 100 most similar rows using mse
            mse = np.ravel(np.power(cached_magnitudes - target_magnitudes, 2).sum(axis=-1))
            sample_idxs = mse.argsort()[:100]
            similar_magnitudes = model_cache[layer_idx].csr_matrix[sample_idxs]

            # Estimate masked feature magnitudes
            group_estimates: dict[Feature, float] = {}
            for feature in masked_feature_group:
                estimate = similar_magnitudes[:, feature.feature_idx].mean()
                group_estimates[feature] = estimate.item()

            # Combine estimates
            estimates.update(group_estimates)

    # print("Estimates:" + " ".join([f"{f}: {round(v, 2)}" for f, v in estimates.items()]))  # DEBUG
    return estimates
