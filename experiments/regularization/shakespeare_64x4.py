"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4 --experiment=0
"""

import argparse
from copy import deepcopy
from typing import Callable

import torch

from config import TrainingConfig
from config.gpt.models import GPTConfig
from config.gpt.training import options as gpt_options
from config.sae.training import options as sae_options
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


def sweep_parameters(fn: Callable, starting_from: tuple[float, ...], ending_with: tuple[float, ...], steps: int):
    """
    Sweep over a range of parameters.
    """
    for i in range(steps):
        coefficients = tuple(start + (end - start) * i / (steps - 1) for start, end in zip(starting_from, ending_with))
        parameter_sets.append({"version": i, "loss_coefficients": coefficients})

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
            trainer = RegularizationTrainer(config, torch.tensor(100.0))
            trainer.train()

        case 2:
            # Load configuration
            config = sae_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "exp.sae.v1.shakespeare_64x4"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.4, 0.8, 7.5, 23.0, 40.0)

            # Initialize trainer
            trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "exp.shakespeare_64x4")
            trainer.train()

        case 3:
            # Train model with specific loss coefficients
            def train_model(version: int, device: torch.device, loss_coefficients: tuple[float, ...]):
                # Load configuration
                config = deepcopy(sae_options["train.b.gated_v2x8.shakespeare_64x4"])
                config.name = f"exp.sae.v2.{version}.shakespeare_64x4"
                config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
                config.loss_coefficients.l1 = loss_coefficients
                config.device = device
                config.compile = False  # Multithreading seems to cause issues with compiling.

                # Initialize trainer
                trainer = ConcurrentTrainer(config, load_from=TrainingConfig.checkpoints_dir / "exp.shakespeare_64x4")
                trainer.device = device
                trainer.model.to(device)
                trainer.train()

            # Sweep over loss coefficients
            parameter_sets = []
            starting_coefficients = (0.15, 0.33, 1.5, 3.8, 4.1)
            ending_coefficients = (0.5, 1.0, 10.0, 32.5, 50.5)
            sweep_parameters(train_model, starting_coefficients, ending_coefficients, steps=2)
