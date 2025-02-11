import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from circuits.features.cache import LayerCache, ModelCache


class ModelProfile:
    """
    Contains feature profiles for a model.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Cached feature profiles for every layer of a model.
        """
        self.layers: dict[int, LayerProfile] = {}

        # Load from checkpoint if provided
        if checkpoint_dir is not None:
            for layer_idx in range(self.get_num_layers(checkpoint_dir)):
                layer_profile = LayerProfile(layer_idx)
                layer_profile.load(checkpoint_dir)
                self.layers[layer_idx] = layer_profile

    def __getitem__(self, layer_idx: int) -> "LayerProfile":
        """
        Get the layer profile for a given layer index.
        """
        return self.layers[layer_idx]

    def compute(self, model_cache: ModelCache):
        """
        Compute feature profiles.
        """
        for layer_idx, layer_cache in tqdm(model_cache.layers.items(), desc="Computing statistics"):
            layer_profile = LayerProfile(layer_idx)
            layer_profile.compute(layer_cache)
            self.layers[layer_idx] = layer_profile

    def save(self, checkpoint_dir: Path):
        """
        Save feature profiles to disk.
        """
        for layer_profile in self.layers.values():
            layer_profile.save(checkpoint_dir)

    def get_num_layers(self, checkpoint_dir: Path) -> int:
        """
        Count the number of files prefixed with "metrics.features."
        """
        if count := len([f for f in checkpoint_dir.iterdir() if f.name.startswith("metrics.features.")]):
            return count
        raise FileNotFoundError("Profiles don't exist. Please run `compute_metrics`.")


class LayerProfile:
    """
    Contains feature profiles for a layer.
    """

    def __init__(self, layer_idx: int):
        """
        Create empty layer profile.
        """
        self.layer_idx = layer_idx
        self.features: dict[int, FeatureProfile] = {}

    def __getitem__(self, feature_idx: int) -> "FeatureProfile":
        """
        Get the feature profile for a given feature index.
        """
        return self.features[feature_idx]

    def compute(self, layer_cache: LayerCache):
        """
        Compute feature profiles.
        """
        # Compute feature profiles
        num_samples = layer_cache.csc_matrix.shape[0]  # type: ignore
        num_features = layer_cache.csc_matrix.shape[-1]  # type: ignore
        for feature_idx in range(num_features):
            non_zero_magnitudes = layer_cache.csc_matrix[:, feature_idx].data

            # Basic statistics
            mean = round(non_zero_magnitudes.mean().astype(float), 4) if non_zero_magnitudes.size > 0 else 0
            std = round(non_zero_magnitudes.std().astype(float), 4) if non_zero_magnitudes.size > 0 else 0
            min = round(non_zero_magnitudes.min().astype(float), 4) if non_zero_magnitudes.size > 0 else 0
            max = round(non_zero_magnitudes.max().astype(float), 4) if non_zero_magnitudes.size > 0 else 0
            count = non_zero_magnitudes.size
            sparsity = round(non_zero_magnitudes.size / num_samples, 4)

            # Create a histogram of the feature magnitudes using 10 bins
            hist, bin_edges = np.histogram(non_zero_magnitudes, bins=10)

            # Store metrics
            self.features[feature_idx] = FeatureProfile(
                mean=mean,
                std=std,
                min=min,
                max=max,
                count=count,
                sparsity=sparsity,
                histogram_counts=hist.tolist(),  # type: ignore
                histogram_edges=[round(x.astype(float), 4) for x in bin_edges],
            )

    def save(self, checkpoint_dir: Path):
        """
        Save feature profiles to disk.
        """
        with open(checkpoint_dir / self.filename, "w") as f:
            data = {k: dataclasses.asdict(v) for k, v in self.features.items()}
            serialized_data = json.dumps(data, indent=2)

            # Regex pattern to remove new lines between "[" and "]"
            pattern = re.compile(r'\[\s*([^"]*?)\s*\]', re.DOTALL)
            serialized_data = pattern.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", serialized_data)

            f.write(serialized_data)

    def load(self, checkpoint_dir: Path):
        """
        Load feature profiles from disk.
        """
        with open(checkpoint_dir / self.filename, "r") as f:
            features_data = json.load(f)
            self.features = {int(k): FeatureProfile(**v) for k, v in features_data.items()}

    @property
    def filename(self) -> str:
        """
        Return the output file path.
        """
        return f"metrics.features.{self.layer_idx}.json"


@dataclass
class FeatureProfile:
    """
    Feature profile containing basic statistics and a histogram.
    """

    mean: float
    std: float
    min: float
    max: float
    count: int
    sparsity: float
    histogram_counts: list[int]
    histogram_edges: list[float]
