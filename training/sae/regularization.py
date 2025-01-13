"""
Train a GPT model with experimental SAE regularization. By adding SAE regularization to GPT training,
we hope to generate GPT model weights are amenable to producing sparser and higher quality SAE features.

$ python -m training.sae.regularization --config=train.b.gated_v2x8.shakespeare_64x4 --name=sparse_shakespeare_64x4
"""

import argparse

import torch

from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training.sae import SAETrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Training config")
    parser.add_argument("--name", type=str, help="Model name for checkpoints")
    return parser.parse_args()


class RegularizationTrainer(SAETrainer):
    """
    Experimental trainer that adds SAE regularization to GPT training.
    """

    λ = 1.0  # Regularization coefficient

    def __init__(self, config: SAETrainingConfig):
        """
        Load new sparsified GPT model from config.
        """
        # create model
        model = SparsifiedGPT(config.sae_config, config.loss_coefficients, config.trainable_layers)

        super().__init__(model, config)

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Add mean SAE loss to GPT cross-entropy loss.
        """
        return output.cross_entropy_loss + self.λ * output.sae_loss


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Update outdir
    config.name = args.name

    # Initialize trainer
    trainer = RegularizationTrainer(config)
    trainer.train()

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
