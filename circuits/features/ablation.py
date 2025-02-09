from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, Sequence

import numpy as np
import torch

from circuits.features import Feature
from circuits.features.cache import ModelCache


class Ablator(Protocol):
    """
    Ablation interface.
    """

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,
        circuit_features: Sequence[Feature],
        num_samples: int,
    ) -> torch.Tensor:
        """
        Patch feature magnitudes and return `num_samples` samples.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param target_token_idx: Target token index for which logits are evaluated.
        :param feature_magnitudes: Feature magnitudes to patch. Shape: (T, F)
        :param circuit_features: Features to preserve.
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, T, F)
        """
        ...


class ResampleAblator(Ablator):
    """
    Ablation using resampling. Based on technique described here:
    https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN
    """

    def __init__(self, model_cache: ModelCache, k_nearest: int = 128):
        """
        :param model_cache: Model cache to use for resampling.
        :param k_nearest: Max number of nearest neighbors to use for creating sample distributions.
        """
        self.model_cache = model_cache
        self.k_nearest = k_nearest

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,  # Shape: (T, F)
        circuit_features: Sequence[Feature],
        num_samples: int,
    ) -> torch.Tensor:  # Shape: (B, T, F)
        """
        Resample feature magnitudes using cached values, returning `num_samples` samples.
        Samples are drawn from `k_nearest` most similar rows in the cache.
        """
        # Construct empty samples
        samples_shape = (num_samples,) + feature_magnitudes.shape
        samples = torch.zeros(samples_shape, device=feature_magnitudes.device)

        # Ignore tokens after the target token because they'll be ignored.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.patch_token_magnitudes,
                    layer_idx,
                    token_idx,
                    feature_magnitudes[token_idx].cpu().numpy(),  # Shape: (F)
                    {f for f in circuit_features if f.token_idx == token_idx},
                    num_samples,
                )
                for token_idx in range(target_token_idx + 1)
            ]

            for future in as_completed(futures):
                token_idx, token_samples = future.result()  # Shape: (B, F)
                samples[:, token_idx, :] = token_samples

        return samples

    def patch_token_magnitudes(
        self,
        layer_idx: int,
        token_idx: int,
        token_feature_magnitudes: np.ndarray,
        token_features: set[Feature],
        num_samples: int,
    ) -> tuple[int, torch.Tensor]:
        """
        Patch feature magnitudes for a single token.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param token_idx: Token index for which features are sampled.
        :param token_feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param token_features: Features to preserve for this token.
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, F)
        """
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size

        # Get circuit features for this token
        feature_idxs = np.array([f.feature_idx for f in token_features])
        num_features = len(feature_idxs)

        if num_features == 0:
            # No features to use for sampling
            top_feature_idxs = np.array([])
        else:
            # Get top features by magnitude
            num_top_features = max(0, min(3, num_features))  # Limit to 3 features
            select_feature_magnitudes = token_feature_magnitudes[feature_idxs]
            # TODO: Consider selecting top features using normalized magnitude
            select_indices = np.argsort(select_feature_magnitudes)[-num_top_features:]
            top_feature_idxs = feature_idxs[select_indices]

        # Find rows in layer cache with any of the top features
        row_idxs = np.unique(layer_cache.csc_matrix[:, top_feature_idxs].nonzero()[0])

        # Use random rows selected based on token position if no feature-based candidates
        if len(row_idxs) == 0:
            num_blocks = layer_cache.magnitudes.shape[0] // block_size  # type: ignore
            sample_idxs = np.random.choice(range(num_blocks), size=num_samples, replace=True)
            row_idxs = sample_idxs * block_size + token_idx

        # Create matrix of token magnitudes to sample from
        candidate_samples = layer_cache.csc_matrix[:, feature_idxs][row_idxs, :].toarray()

        # Add column for token positional distance
        positional_distances = np.abs((row_idxs % block_size) - token_idx).astype(np.float32)
        positional_distances = positional_distances / block_size  # Scale to [0, 1]
        candidate_samples = np.column_stack((candidate_samples, positional_distances))

        # Calculate MSE
        target_values = token_feature_magnitudes[feature_idxs]
        target_values = np.append(target_values, 0)  # Append positional distance of 0
        mse = np.mean((candidate_samples - target_values) ** 2, axis=-1)

        # Get nearest neighbors
        num_neighbors = min(self.k_nearest, len(row_idxs))
        # TODO: Consider removing first exact match to avoid duplicating original values
        nearest_neighbor_idxs = row_idxs[np.argsort(mse)[:num_neighbors]]

        # Randomly draw indices from top candidates (with replacement) to include in samples
        sample_idxs = np.random.choice(nearest_neighbor_idxs, size=num_samples, replace=True)
        token_samples = torch.tensor(layer_cache.csr_matrix[sample_idxs, :].toarray())  # Shape: (B, F)
        return token_idx, token_samples


class ZeroAblator(Ablator):
    """
    Ablation using zeroing of patched features.
    """

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,
        circuit_features: Sequence[Feature],
        num_samples: int,
    ) -> torch.Tensor:
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
