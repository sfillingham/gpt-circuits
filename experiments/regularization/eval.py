"""
Evaluate a GPT model that has been trained with SAE regularization.

$ python -m experiments.regularization.eval --step=0
"""

import argparse
import shutil

from config import TrainingConfig
from config.gpt.models import gpt_options
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import (
    LossCoefficients,
    SAETrainingConfig,
    shakespeare_64x4_defaults,
)
from training.sae.concurrent import ConcurrentTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Which step to run")
    return parser.parse_args()


def create_training_config(
    name: str,
    l1_coefficients: tuple[float, ...],
) -> SAETrainingConfig:
    """
    Create configuration to be used for SAE training.
    """
    config = SAETrainingConfig(
        name=name,
        sae_config=SAEConfig(
            gpt_config=gpt_options["ascii_64x4"],
            n_features=tuple(64 * n for n in (8, 8, 32, 64, 64)),
            sae_variant=SAEVariant.STANDARD,
        ),
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=l1_coefficients,
        ),
    )

    return config


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    base_dir = TrainingConfig.checkpoints_dir / "regularization"

    match args.step:
        case 0:
            """
            Copy model weights from prior experiment.
            """
            experiment_name = "layers"

            # Copy normal model
            src_dir = base_dir / f"{experiment_name}.model.normal"
            dst_dir = base_dir / "eval.model.normal"
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

            # Copy regularized model
            src_dir = base_dir / f"{experiment_name}.model.regularized"
            dst_dir = base_dir / "eval.model.regularized"
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        case 1:
            """
            Train SAE layers on GPT model with normal weights.
            """
            config = create_training_config(
                name="eval.saes.normal",
                l1_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # TODO: Set L1 coefficients
            )

            # Train model
            trainer = ConcurrentTrainer(config, load_from=base_dir / "eval.model.normal")
            trainer.train()

        case 2:
            """
            Train SAE layers on GPT model with regularized weights.
            """
            config = create_training_config(
                name="eval.saes.regularized",
                l1_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # TODO: Set L1 coefficients
            )

            # Train model
            trainer = ConcurrentTrainer(config, load_from=base_dir / "eval.model.regularized")
            trainer.train()

        case 3:
            """
            Evaluate SAE differences.
            """
            pass
