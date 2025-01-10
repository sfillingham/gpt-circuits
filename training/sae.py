"""
Train GPT model:
$ python -m training.sae --config=gated_v2_shakespeare_64x4

DDP launch for e.g. 8 GPUs:
$ torchrun --standalone --nproc_per_node=8 -m training.sae --config=gated_v2_shakespeare_64x4
"""

import argparse
from typing import Optional

import torch

from config.sae.training import SAETrainingConfig, options
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training import Trainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gated_v2_shakespeare_64x4", help="Training config")
    parser.add_argument("--load_from", type=str, help="Path to load model from")
    return parser.parse_args()


class SAETrainer(Trainer):
    """
    Trainer for GPT models.
    """

    def __init__(self, config: SAETrainingConfig, load_from: str | None = None):
        """
        Load GPT model.
        """
        # create model
        if load_from:
            model = SparsifiedGPT.load(load_from, config.loss_coefficients, device=config.device)
            print(f"Loaded saved model from {args.load_from}")
        else:
            model = SparsifiedGPT(config.sae_config, config.loss_coefficients, config.trainable_layers)

        super().__init__(model, config)

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Calculate model loss.
        """
        output: SparsifiedGPTOutput = self.model(x, y)
        loss = output.cross_entropy_loss + output.sae_loss
        metrics = None

        # Only include metrics if in evaluation mode
        if is_eval:
            metrics = {
                "ce_loss": f"{output.cross_entropy_loss:.4f}",
                "sae_loss": f"{output.sae_loss:.4f}",
            }

        return loss, metrics


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Initialize trainer
    trainer = SAETrainer(config, args.load_from)
    trainer.train()

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
