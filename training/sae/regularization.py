"""
Train a GPT model with experimental SAE regularization. By adding SAE regularization to GPT training,
we hope to generate GPT model weights are amenable to producing sparser and higher quality SAE features.

$ python -m training.sae.regularization --config=standardx8.shakespeare_64x4.v1 --name=shakespeare_64x4.regularized
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

    λ: torch.Tensor  # Regularization coefficient (applied to reconstruction loss)
    checkpoint_ce_loss: torch.Tensor  # Cross-entropy loss for saved checkpoint

    def __init__(self, config: SAETrainingConfig, λ=torch.tensor(1.0)):
        """
        Load new sparsified GPT model from config.
        """
        assert config.loss_coefficients.regularization is not None
        self.λ = config.loss_coefficients.regularization.to(config.device)

        # create model
        model = SparsifiedGPT(config.sae_config, config.loss_coefficients, config.trainable_layers)
        self.checkpoint_ce_loss = torch.tensor(float("inf"), device=config.device)

        super().__init__(model, config)

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Add SAE loss components to GPT cross-entropy loss.
        """
        # Scale losses based on MSE by input norms to avoid weight shrinkage during training.
        x_norms = torch.stack([loss.x_norm for loss in output.sae_loss_components.values()])
        reconstruct_losses = torch.stack([loss.reconstruct for loss in output.sae_loss_components.values()]) / x_norms
        aux_losses = torch.stack([loss.aux for loss in output.sae_loss_components.values()]) / x_norms

        # Sparcity loss doesn't scale with input norms.
        sparcity_losses = torch.stack([loss.sparsity for loss in output.sae_loss_components.values()])

        # Only scale MSE-based losses by λ when computing the regularization term.
        regularization_term = (reconstruct_losses * self.λ + sparcity_losses + aux_losses * self.λ).mean()

        return output.cross_entropy_loss + regularization_term

    def save_checkpoint(self, model: SparsifiedGPT, is_best: torch.Tensor, metrics: dict[str, torch.Tensor]):
        """
        Save CE loss when saving checkpoint.
        """
        super().save_checkpoint(model, is_best, metrics)
        self.checkpoint_ce_loss = metrics["ce_loss"]


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
