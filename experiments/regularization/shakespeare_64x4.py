"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4 --experiment=0
"""

import argparse
from pathlib import Path

import torch

from config import TrainingConfig
from config.gpt.models import GPTConfig, gpt_options
from config.gpt.training import options as gpt_training_options
from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import (
    LossCoefficients,
    SAETrainingConfig,
    shakespeare_64x4_defaults,
)
from config.sae.training import options as sae_traing_options
from experiments import ParameterSweeper
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


def sweep_training_parameters(
    name_prefix: str, load_from: Path, starting_from: tuple[float, ...], ending_with: tuple[float, ...], steps: int
):
    """
    Sweep over a range of parameters.
    """
    for i in range(steps):
        coefficients = tuple(start + (end - start) * i / (steps - 1) for start, end in zip(starting_from, ending_with))
        parameter_sets.append(
            {
                "name": f"{name_prefix}{i}",
                "load_from": load_from,
                "loss_coefficients": coefficients,
            }
        )

    # Assign devices
    devices = (
        [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [GPTConfig(name="").device]
    )
    for i, parameters in enumerate(parameter_sets):
        parameters["device"] = devices[i % len(devices)]

    # Run parameter sweep
    sweeper = ParameterSweeper(train_model, parameter_sets, pool_size=len(devices))
    sweeper.sweep()


def train_model(name: str, load_from: Path, device: torch.device, loss_coefficients: tuple[float, ...]):
    """
    Train a model with specific loss coefficients.
    """
    # Load configuration
    config = SAETrainingConfig(
        name=name,
        device=device,
        sae_config=SAEConfig(
            gpt_config=gpt_options["ascii_64x4"],
            n_features=tuple(64 * n for n in (8, 8, 32, 64, 64)),
            sae_variant=SAEVariant.GATED_V2,
        ),
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=loss_coefficients,
        ),
    )

    # Train model
    trainer = ConcurrentTrainer(config, load_from=load_from)
    trainer.train()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    match args.experiment:
        case 0:
            # Load configuration
            config = gpt_training_options["shakespeare_64x4"]
            config.name = "exp.shakespeare_64x4"

            # Initialize trainer
            trainer = GPTTrainer(config)
            trainer.train()

        case 1:
            # Load configuration
            config = sae_traing_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sparse_shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 15000

            # Initialize trainer
            trainer = RegularizationTrainer(config, torch.tensor(100.0))
            trainer.train()

        case 2:
            # Load configuration
            config = sae_traing_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v1.shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.4, 0.8, 7.5, 23.0, 40.0)

            # Initialize trainer
            trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "exp.shakespeare_64x4")
            trainer.train()

        case 3:
            # Sweep over loss coefficients
            parameter_sets = []
            starting_coefficients = (0.15, 0.33, 1.5, 3.8, 4.1)
            ending_coefficients = (0.5, 1.0, 10.0, 32.5, 50.5)
            sweep_training_parameters(
                name_prefix="exp.sae.v2.shakespeare_64x4.",
                load_from=TrainingConfig.checkpoints_dir / "exp.shakespeare_64x4",
                starting_from=starting_coefficients,
                ending_with=ending_coefficients,
                steps=2,
            )
