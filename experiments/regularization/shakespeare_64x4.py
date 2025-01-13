"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4
"""

import argparse

from config.sae.training import options
from training.sae.regularization import RegularizationTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="exp.sparse_shakespeare_64x4", help="Model name for checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = options["train.a.gated_v2x8.shakespeare_64x4"]
    config.name = args.name
    config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
    config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
    config.max_steps = 10000

    # Initialize trainer
    trainer = RegularizationTrainer(config, 100.0)
    trainer.train()

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
