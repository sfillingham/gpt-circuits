from dataclasses import dataclass

import numpy as np

from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile


@dataclass(frozen=True)
class ClusterCacheKey:
    """
    Cache key for clustering results.
    """

    layer_idx: int
    token_idx: int
    circuit_feature_idxs: tuple[int, ...]
    circuit_feature_magnitudes: tuple[float, ...]
    k_nearest: int
    positional_coefficient: float


class ClusterSearch:
    """
    Search for nearest neighbors using cached feature magnitudes.
    """

    # Map of cached nearest neighbors
    nearest_neighbors_cache: dict[ClusterCacheKey, np.ndarray] = {}

    def __init__(self, model_profile: ModelProfile, model_cache: ModelCache):
        self.model_profile = model_profile
        self.model_cache = model_cache

    def get_nearest_neighbors(
        self,
        layer_idx: int,
        token_idx: int,
        feature_magnitudes: np.ndarray,  # Shape: (F)
        circuit_feature_idxs: np.ndarray,
        k_nearest: int,
        positional_coefficient: float,
    ) -> np.ndarray:
        """
        Get nearest neighbors for a single token in a given circuit.

        :param layer_cache: Layer cache to use for resampling.
        :param token_idx: Token index for which features are sampled.
        :param feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param circuit_feature_idxs: Indices of features to preserve for this token.

        :return: Indices of nearest neighbors.
        """
        assert k_nearest > 0
        layer_profile = self.model_profile[layer_idx]
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size
        num_features = len(circuit_feature_idxs)

        # Check if nearest neighbors are cached
        # TODO: Consider purging unused cache entries
        circuit_feature_magnitudes = feature_magnitudes[circuit_feature_idxs]
        cache_key = self.get_cache_key(
            layer_idx,
            token_idx,
            circuit_feature_idxs,
            circuit_feature_magnitudes,
            k_nearest,
            positional_coefficient,
        )
        if cache_key in self.nearest_neighbors_cache:
            return self.nearest_neighbors_cache[cache_key]

        if num_features == 0:
            # No features to use for sampling
            top_feature_idxs = np.array([])
        else:
            # Get top features by magnitude
            num_top_features = max(0, min(16, num_features))  # Limit to 16 features
            # TODO: Consider selecting top features using normalized magnitude
            select_indices = np.argsort(circuit_feature_magnitudes)[-num_top_features:]
            top_feature_idxs = circuit_feature_idxs[select_indices]

        # Find rows in layer cache with any of the top features
        row_idxs = np.unique(layer_cache.csc_matrix[:, top_feature_idxs].nonzero()[0])

        # Use random rows selected based on token position if no feature-based candidates
        if len(row_idxs) == 0:
            num_blocks = layer_cache.magnitudes.shape[0] // block_size  # type: ignore
            sample_idxs = np.random.choice(range(num_blocks), size=k_nearest, replace=True)
            row_idxs = sample_idxs * block_size + token_idx

        # Create matrix of token magnitudes to sample from
        candidate_samples = layer_cache.csc_matrix[:, circuit_feature_idxs][row_idxs, :].toarray()
        target_values = feature_magnitudes[circuit_feature_idxs]

        # Calculate normalization coefficients
        norm_coefficients = np.ones_like(target_values)
        for i, feature_idx in enumerate(circuit_feature_idxs):
            feature_profile = layer_profile[int(feature_idx)]
            norm_coefficients[i] = 1.0 / feature_profile.max

        # Add positional information
        positional_distances = np.abs((row_idxs % block_size) - token_idx).astype(np.float32)
        positional_distances = positional_distances / block_size  # Scale to [0, 1]
        candidate_samples = np.column_stack((candidate_samples, positional_distances))  # Add column
        target_values = np.append(target_values, 0)  # Add target
        norm_coefficients = np.append(norm_coefficients, positional_coefficient)  # Add coefficient

        # Calculate MSE
        errors = (candidate_samples - target_values) * norm_coefficients
        mse = np.mean(errors**2, axis=-1)

        # Get nearest neighbors
        num_neighbors = min(k_nearest, len(row_idxs))
        # TODO: Consider removing first exact match to avoid duplicating original values
        nearest_neighbor_idxs = row_idxs[np.argsort(mse)[:num_neighbors]]

        # Cache nearest neighbors
        self.nearest_neighbors_cache[cache_key] = nearest_neighbor_idxs
        return nearest_neighbor_idxs

    def get_cache_key(
        self,
        layer_idx: int,
        token_idx: int,
        circuit_feature_idxs: np.ndarray,
        circuit_feature_magnitudes: np.ndarray,
        k_nearest: int,
        positional_coefficient: float,
    ) -> ClusterCacheKey:
        """
        Get cache key for nearest neighbors.
        """
        return ClusterCacheKey(
            layer_idx,
            token_idx,
            tuple([int(f) for f in circuit_feature_idxs]),
            tuple([float(f) for f in circuit_feature_magnitudes]),
            k_nearest,
            positional_coefficient,
        )
