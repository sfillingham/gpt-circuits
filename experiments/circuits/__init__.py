import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from circuits.features.cache import LayerCache, ModelCache
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
        model_cache[layer_idx], feature_magnitudes, circuit_features, target_token_idx, num_samples=16
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
    for feature in tqdm(circuit_features, desc="Estimating ablation effects"):
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
    layer_cache: LayerCache,
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_features: Sequence[Feature],
    target_token_idx: int,
    num_samples: int = 16,
) -> torch.Tensor:  # Shape: (B, T, F)
    """
    Resample feature magnitudes using cached values, returning N samples.
    Samples are drawn from the top 100 most similar rows in the cache.

    Based on technique described here: https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN
    """
    # Construct empty samples
    block_size = feature_magnitudes.size(0)
    samples_shape = (num_samples, block_size, feature_magnitudes.size(-1))
    samples = torch.zeros(samples_shape, device=feature_magnitudes.device)

    # Ignore tokens after the target token because they'll be ignored.
    for token_idx in range(target_token_idx + 1):
        # Get circuit features for this token
        token_feature_magnitudes = feature_magnitudes[token_idx].cpu()  # Shape: (F)
        feature_idxs = torch.tensor([f.feature_idx for f in circuit_features if f.token_idx == token_idx])
        num_features = feature_idxs.size(0)

        if num_features == 0:
            # No features to use for sampling
            top_feature_idxs = torch.tensor([])
        else:
            # Get top features by magnitude
            num_top_features = max(0, min(3, num_features))  # Limit to 3 features
            select_feature_magnitudes = token_feature_magnitudes[feature_idxs]
            # TODO: Consider selecting top features using normalized magnitude
            select_indices = torch.topk(select_feature_magnitudes, k=num_top_features).indices
            top_feature_idxs = feature_idxs[select_indices]

        # Find rows in layer cache with any of the top features
        top_column_idxs = top_feature_idxs.numpy()
        row_idxs = np.unique(layer_cache.csc_matrix[:, top_column_idxs].nonzero()[0])

        # Use random rows selected based on token position if no feature-based candidates
        if len(row_idxs) == 0:
            num_blocks = layer_cache.magnitudes.shape[0] // block_size  # type: ignore
            sample_idxs = np.random.choice(range(num_blocks), size=num_samples, replace=True)
            row_idxs = sample_idxs * block_size + token_idx

        # Create matrix of token magnitudes to sample from
        column_idxs = feature_idxs.numpy()
        candidate_samples = layer_cache.csc_matrix[:, column_idxs][row_idxs, :].toarray()

        # Add column for token positional distance
        positional_distances = np.abs((row_idxs % block_size) - token_idx).astype(np.float32)
        positional_distances = positional_distances / block_size  # Scale to [0, 1]
        candidate_samples = np.column_stack((candidate_samples, positional_distances))

        # Calculate MSE
        target_values = token_feature_magnitudes[column_idxs].numpy()
        target_values = np.append(target_values, 0)  # Append positional distance of 0
        mse = np.mean((candidate_samples - target_values) ** 2, axis=-1)

        # Get top candidates
        num_candidates = min(100, len(row_idxs))
        # TODO: Consider removing first exact match to avoid duplicating original values
        top_candidate_idxs = row_idxs[np.argsort(mse)[:num_candidates]]

        # Randomly draw indices from top candidates (with replacement) to include in samples
        sample_idxs = np.random.choice(top_candidate_idxs, size=num_samples, replace=True)
        token_samples = torch.tensor(layer_cache.csr_matrix[sample_idxs, :].toarray())  # Shape: (B, F)
        samples[:, token_idx, :] = token_samples

    return samples


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
