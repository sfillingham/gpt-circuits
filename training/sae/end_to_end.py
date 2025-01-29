"""
Train SAE weights using "End-to-End Sparse Dictionary Learning" for all layers concurrently.

$ python -m training.sae.end_to_end --config=standardx8.shakespeare_64x4.v0 --load_from=shakespeare_64x4 --name=sae.shakespeare_64x4.v0
"""

import argparse

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
        Return a loss for each trainable layer.

        TODO: Implement end-to-end sparse dictionary learning.
        """
        return output.sae_losses


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
