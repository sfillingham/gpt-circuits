"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4 --experiment=0
"""

import argparse

from config import TrainingConfig
from config.gpt.training import options as gpt_options
from config.sae.training import options as sae_options
from training.gpt import GPTTrainer
from training.sae.concurrent import ConcurrentTrainer
from training.sae.regularization import RegularizationTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=0, help="Which experiment to run")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    match args.experiment:
        case 0:
            # Load configuration
            config = gpt_options["shakespeare_64x4"]
            config.name = "exp.shakespeare_64x4"

            # Initialize trainer
            trainer = GPTTrainer(config)
            trainer.train()

        case 1:
            # Load configuration
            config = sae_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sparse_shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 15000

            # Initialize trainer
            trainer = RegularizationTrainer(config, 100.0)
            trainer.train()

        case 2:
            # Load configuration
            config = sae_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v1.shakespeare_64x4"
            config.loss_coefficients.l1 = (0.5, 0.7, 1.1, 2.2, 3.2)

            # Initialize trainer
            trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "exp.shakespeare_64x4")
            trainer.train()

        case 3:
            # Load configuration
            config = sae_options["train.b.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v2.shakespeare_64x4"
            config.loss_coefficients.l1 = (0.3, 0.3, 0.3, 0.3, 0.3)

            # Initialize trainer
            trainer = ConcurrentTrainer(
                config, load_from=TrainingConfig.checkpoints_dir / "exp.sparse_shakespeare_64x4"
            )
            trainer.train()
