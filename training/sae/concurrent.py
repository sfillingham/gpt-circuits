"""
Train SAE weights for all layers concurrently.

$ python -m training.sae.concurrent --config=gated_v2_shakespeare_64x4 --load_from=shakespeare_64x4 --name=sae.v1.shakespeare_64x4
"""

import argparse

from config import TrainingConfig
from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPT
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
