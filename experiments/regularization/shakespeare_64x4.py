"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4
"""

import argparse

from config import TrainingConfig
from config.sae.training import options
from training.sae.concurrent import ConcurrentTrainer
from training.sae.regularization import RegularizationTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=1, help="Which experiment to run")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    match args.experiment:
        case 1:
            # Load configuration
            config = options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sparse_shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 12000

            # Initialize trainer
            trainer = RegularizationTrainer(config, 100.0)
            trainer.train()

        case 2:
            # Load configuration
            config = options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v1.shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 12000

            # Initialize trainer
            trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "shakespeare_64x4")
            trainer.train()

        case 3:
            # Load configuration
            config = options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v2.shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 12000

            # Initialize trainer
            trainer = ConcurrentTrainer(
                config, load_from=TrainingConfig.checkpoints_dir / "exp.sparse_shakespeare_64x4"
            )
            trainer.train()
