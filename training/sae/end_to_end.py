"""
Train SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.

$ python -m training.sae.end_to_end --config=standardx8.shakespeare_64x4.v0 --load_from=shakespeare_64x4 --name=sae.shakespeare_64x4.v0
"""

import argparse
from contextlib import contextmanager
from typing import Iterable

import torch

from config import TrainingConfig
from config.sae.training import options
from models.sparsified import SparsifiedGPTOutput
from training.sae.concurrent import ConcurrentTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--load_from", type=str, help="GPT model weights to load")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class EndToEndTrainer(ConcurrentTrainer):
    """
    Train SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.
    https://arxiv.org/pdf/2405.12241
    """

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Return a stack of end-to-end losses, one for each trainable layer.
        """
        losses = []
        for layer_idx, reconstructed_activations in output.reconstructed_activations.items():
            losses.append(self.calculate_e2e_loss(layer_idx, reconstructed_activations, output.logits))

        return torch.stack(losses)

    def calculate_e2e_loss(
        self, layer_idx: int, reconstructed_activations: torch.Tensor, target_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate end-to-end loss for a single layer.

        :param layer_idx: Index of the layer to calculate loss for.
        :param reconstructed_activations: Reconstructed residual stream activations at this layer.
        :param target: Target logits.
        """
        vocab_size = target_logits.size(-1)
        n_blocks = len(self.model.gpt.transformer.h)
        target_layers = list(range(layer_idx + 1, n_blocks))

        # Forward pass with reconstructed activations
        with self.collect_activations(target_layers=target_layers) as activations:
            reconstructed_logits = self.model.gpt.forward_with_patched_activations(
                reconstructed_activations, layer_idx
            )

        # TODO: Caculate downstream reconstruction loss

        # Calculate KL divergence between target and reconstructed logits
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(reconstructed_logits, dim=-1).view(-1, vocab_size),
            torch.nn.functional.softmax(target_logits, dim=-1).view(-1, vocab_size),
            reduction="batchmean",
        )
        return kl_div

    @contextmanager
    def collect_activations(self, target_layers: Iterable[int] = ()):
        """
        Context manager for collecting activations during the forward pass.

        :param target_layers: Target layer indices. 1 means we collect activations just after the first transformer block.
        :yield activations: Dictionary of activations.
        """
        # Dictionary for storing results
        activations: dict[int, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in target_layers:
            assert layer_idx > 0, "Must target activations after the first transformer block"
            target = self.model.gpt.transformer.h[layer_idx - 1]
            # TODO: Implement hook

        try:
            yield activations

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Update outdir
    config.name = args.name

    # Initialize trainer
    trainer = EndToEndTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()
