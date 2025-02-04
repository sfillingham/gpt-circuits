"""
Cache and load feature metrics.
"""

import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from config.sae.models import SAEConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class FeatureSet:
    SAMPLES_PERCENTILE = 90  # Percentile to use for feature samples
    SUGGESTED_SAMPLE_COUNT = 100  # Number of samples to use (if possible)

    def __init__(self, checkpoint_dir: Path):
        """
        Load feature metrics from cache.
        """
        self.checkpoint_dir = checkpoint_dir

        # Check that files starting with "features_" exist
        has_metrics = any(f.startswith("metrics.features.") for f in os.listdir(checkpoint_dir))
        if not has_metrics:
            raise ValueError(f"No metrics found in {checkpoint_dir}. Please run `FeatureSet.cache_metrics(...)`.")

        # Load SAE config
        meta_path = checkpoint_dir / "sae.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        config = SAEConfig(**meta)

    @classmethod
    def cache_metrics(cls, model: SparsifiedGPT, checkpoint_dir: Path, shard: DatasetShard, batch_size: int):
        """
        Cache feature metrics.
        """
        batch_feature_magnitudes: dict[int, list[sparse.csr_matrix]] = defaultdict(list)

        # Walk through shard.tokens in batches
        block_size = model.config.block_size
        tokens_per_batch = batch_size * block_size
        num_blocks = len(shard.tokens) // block_size
        for i in tqdm(
            range(0, num_blocks * block_size, tokens_per_batch),
            total=math.ceil(num_blocks / batch_size),
            desc="Extracting feature magnitudes",
        ):
            # Get batch of tokens
            ending_idx = min(shard.tokens.size(-1), i + tokens_per_batch)
            ending_idx -= ending_idx % block_size
            tokens = shard.tokens[i:ending_idx].to(model.config.device)
            tokens = tokens.view(-1, block_size)

            # Get feature magnitudes
            with torch.no_grad():
                output: SparsifiedGPTOutput = model(tokens)
            for layer_idx, feature_magnitudes in output.feature_magnitudes.items():
                # Remove batch dimension
                flattened_feature_magnitudes = feature_magnitudes.view(-1, feature_magnitudes.size(-1))
                # Append sparse matrix
                sparse_matrix = sparse.csr_matrix(flattened_feature_magnitudes.cpu().numpy())
                batch_feature_magnitudes[layer_idx].append(sparse_matrix)

        # Save feature magnitudes and process metrics
        for layer_idx, feature_magnitudes_list in tqdm(batch_feature_magnitudes.items(), desc="Processing layers"):
            # Concatenate sparse matrices
            coo_feature_magnitudes: sparse.coo_matrix = sparse.vstack(feature_magnitudes_list, format="coo")  # type: ignore
            csc_feature_magnitudes: sparse.csc_matrix = coo_feature_magnitudes.tocsc()  # type: ignore # Faster column slicing

            # Save magnitudes to disk
            cls.save_magnitudes(coo_feature_magnitudes, checkpoint_dir, layer_idx)

            # Save statistics to disk
            cls.save_statistics(csc_feature_magnitudes, checkpoint_dir, layer_idx)

            # Save samples to disk
            cls.save_samples(model, shard, csc_feature_magnitudes, checkpoint_dir, layer_idx)

    @classmethod
    def save_magnitudes(cls, coo_feature_magnitudes: sparse.coo_matrix, checkpoint_dir: Path, layer_idx: int):
        """
        Save feature magnitudes to disk.
        """
        # Save feature magnitudes to disk (using COO format)
        npz_path = checkpoint_dir / f"metrics.magnitudes.{layer_idx}.npz"
        sparse.save_npz(npz_path, coo_feature_magnitudes)

    @classmethod
    def save_statistics(cls, csc_feature_magnitudes: sparse.csc_matrix, checkpoint_dir: Path, layer_idx: int):
        """
        Save feature statistics to disk.
        """
        feature_metrics: dict[int, dict] = {}
        num_samples = csc_feature_magnitudes.shape[0]  # type: ignore
        num_features = csc_feature_magnitudes.shape[-1]  # type: ignore
        for feature_idx in range(num_features):
            non_zero_magnitudes = csc_feature_magnitudes[:, feature_idx].data

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
            feature_metrics[feature_idx] = {
                "mean": mean,
                "std": std,
                "min": min,
                "max": max,
                "count": count,
                "sparsity": sparsity,
                "histogram": {
                    "counts": hist.tolist(),
                    "edges": [round(x.astype(float), 4) for x in bin_edges],
                },
            }

        # Save metrics to disk
        metrics_path = checkpoint_dir / f"metrics.features.{layer_idx}.json"
        with open(metrics_path, "w") as f:
            serialized_data = json.dumps(feature_metrics, indent=2)

            # Regex pattern to remove new lines between "[" and "]"
            pattern = re.compile(r"\[\s*(.*?)\s*\]", re.DOTALL)
            serialized_data = pattern.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", serialized_data)

            f.write(serialized_data)

    @classmethod
    def save_samples(
        cls,
        model: SparsifiedGPT,
        shard: DatasetShard,
        csc_feature_magnitudes: sparse.csc_matrix,
        checkpoint_dir: Path,
        layer_idx: int,
    ):
        """
        Save feature samples to disk.
        """
        all_sample_indicies: dict[int, np.ndarray] = {}  # Indices with shape (num_samples,)
        all_sample_magnitudes: dict[int, sparse.csr_matrix] = {}  # Magnitudes with shape (num_samples, block_size)

        num_features = csc_feature_magnitudes.shape[-1]  # type: ignore
        for feature_idx in range(num_features):
            # List of token indices to use for the feature sample
            token_idxs: list[int]

            # Are there enough samples to use percentile thresholding?
            sparse_feature_magnitudes = csc_feature_magnitudes[:, feature_idx]
            non_zero_rows = sparse_feature_magnitudes.nonzero()[0]
            non_zero_magnitudes = sparse_feature_magnitudes.data
            use_topk = non_zero_magnitudes.size < (100 * 100) / (100.0 - cls.SAMPLES_PERCENTILE)
            if use_topk:
                topk = min(non_zero_magnitudes.size, cls.SUGGESTED_SAMPLE_COUNT)
                top_rows = sorted(zip(non_zero_rows, non_zero_magnitudes), key=lambda x: -x[1])[:topk]
                token_idxs = [int(idx) for idx, _ in top_rows]
            else:
                token_idxs = []
                threshold = np.percentile(non_zero_magnitudes, cls.SAMPLES_PERCENTILE)
                top_rows = [(idx, m) for (idx, m) in zip(non_zero_rows, non_zero_magnitudes) if m >= threshold]
                select_rows = sorted(random.sample(top_rows, cls.SUGGESTED_SAMPLE_COUNT), key=lambda x: -x[1])
                token_idxs = [int(idx) for idx, _ in select_rows]

            # Convert token indices to sample index ranges
            block_size = model.config.block_size
            sample_starting_idxs = [idx - idx % block_size for idx in token_idxs]
            sample_ending_idxs = [idx + block_size for idx in sample_starting_idxs]

            # Collect magnitudes for each sample
            sample_feature_magnitudes = []
            for starting_idx, ending_idx in zip(sample_starting_idxs, sample_ending_idxs):
                sparse_feature_magnitudes = csc_feature_magnitudes[starting_idx:ending_idx, feature_idx]
                sparse_feature_magnitudes = sparse_feature_magnitudes.todense().reshape(1, -1)
                sample_feature_magnitudes.append(sparse_feature_magnitudes)

            # Store samples
            if len(sample_feature_magnitudes) > 0:
                sample_feature_magnitudes = sparse.csr_matrix(np.vstack(sample_feature_magnitudes))
                all_sample_indicies[feature_idx] = np.array(sample_starting_idxs)
                all_sample_magnitudes[feature_idx] = sample_feature_magnitudes

        # Save samples to disk
        samples_path = checkpoint_dir / f"metrics.samples.{layer_idx}.npz"
        np.savez(
            samples_path,
            split=shard.split,
            shard_idx=shard.shard_idx,
            **{f"{feature_idx}.indices": v for feature_idx, v in all_sample_indicies.items()},  # type: ignore
            **{f"{feature_idx}.magnitudes": v for feature_idx, v in all_sample_magnitudes.items()},  # type: ignore
        )
