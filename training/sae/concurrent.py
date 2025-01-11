"""
Train SAE weights for all layers concurrently.

$ python -m training.sae.concurrent --config=gated_v2_shakespeare_64x4 --load_from=shakespeare_64x4 --name=sae.v1.shakespeare_64x4
"""

import argparse

import torch

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training.sae import SAETrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gated_v2_shakespeare_64x4", help="Training config")
    parser.add_argument("--load_from", type=str, default="shakespeare_64x4", help="GPT model weights to load")
    parser.add_argument("--name", type=str, default="sae.v1.shakespeare_64x4", help="Model name for checkpoints")
    return parser.parse_args()


class ConcurrentTrainer(SAETrainer):
    """
    Train SAE weights for all layers concurrently.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str):
        """ """
        # Create model
        model = SparsifiedGPT(config.sae_config, config.loss_coefficients, config.trainable_layers)

        # Load GPT weights
        model.load_gpt_weights(load_from)

        # Freeze GPT parameters
        for param in model.gpt.parameters():
            param.requires_grad = False

        super().__init__(model, config)

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Return an array of losses instead of a combined loss.
        """
        reconstruct_losses = torch.stack([loss.reconstruct for loss in output.sae_loss_components.values()])
        sparsity_losses = torch.stack([loss.sparsity for loss in output.sae_loss_components.values()])
        aux_losses = torch.stack([loss.aux for loss in output.sae_loss_components.values()])
        return reconstruct_losses + sparsity_losses + aux_losses

    def backward(self, loss):
        """
        Go through each layer's loss and call backward() on it.
        """
        for i, layer_loss in enumerate(loss):
            is_last_layer = i == len(loss) - 1
            layer_loss.backward(retain_graph=not is_last_layer)

    def save_checkpoint(self, model: SparsifiedGPT, is_best: torch.Tensor):
        """
        Only save SAE weights for layers that have achieved a better validation loss.
        """
        # `is_best` contains a value for each layer indicating whether we have the best loss for that layer.
        layers_to_save = [layer_name for should_save, layer_name in zip(is_best, model.saes.keys()) if should_save]
        model.save(self.config.out_dir, layers_to_save)
        print(f"Saved SAE weights for layers: {', '.join(layers_to_save)}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Update outdir
    config.name = args.name

    # Initialize trainer
    trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / args.load_from)
    trainer.train()

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
