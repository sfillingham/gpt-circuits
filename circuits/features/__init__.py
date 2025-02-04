"""
Utility functions for computing and caching feature metrics.
"""

from pathlib import Path

from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.features.samples import ModelSampleSet
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT


def compute_metrics(
    model: SparsifiedGPT,
    checkpoint_dir: Path,
    shard: DatasetShard,
    batch_size: int = 256,
) -> None:
    """
    Compute and cache feature metrics for a given model and dataset.

    :param model: The model to analyze.
    :param checkpoint_dir: The output directory for caching metrics.
    :param shard: The dataset shard to analyze.
    :param batch_size: The batch size to use for analysis.
    """
    model_cache = ModelCache()
    model_cache.compute(model, shard, batch_size)
    model_cache.save(checkpoint_dir)

    model_profile = ModelProfile()
    model_profile.compute(model_cache)
    model_profile.save(checkpoint_dir)

    model_sample_set = ModelSampleSet()
    model_sample_set.compute(model_cache)
    model_sample_set.save(checkpoint_dir)
