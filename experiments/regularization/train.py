"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.train --step=0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd
import torch

from config import TrainingConfig
from config.gpt.models import GPTConfig, gpt_options
from config.gpt.training import options as gpt_training_options
from config.sae.models import SAEConfig
from config.sae.training import (
    LossCoefficients,
    SAETrainingConfig,
    shakespeare_64x4_defaults,
)
from experiments import ParameterSweeper
from experiments.regularization.setup import (  # noqa: F401
    GatedExperimentSetup,
    GatedV2ExperimentSetup,
    StandardExperimentSetup,
    StandardV2ExperimentSetup,
)
from training.gpt import GPTTrainer
from training.sae.concurrent import ConcurrentTrainer
from training.sae.regularization import RegularizationTrainer

# Experiment setups are in setup.py
setup = StandardV2ExperimentSetup()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Which step to run")
    return parser.parse_args()


def create_config(
    name: str,
    loss_coefficients: tuple[float, ...],
    device: torch.device | None = None,
    max_steps: int | None = None,
) -> SAETrainingConfig:
    """
    Create configuration to be used for SAE training or GPT regularization.
    """
    config = SAETrainingConfig(
        name=name,
        sae_config=SAEConfig(
            gpt_config=gpt_options["ascii_64x4"],
            n_features=setup.n_features,
            sae_variant=setup.sae_variant,
        ),
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=loss_coefficients,
        ),
    )

    # Set optional args
    config.device = device or config.device
    config.max_steps = max_steps or config.max_steps

    return config


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
                "name": f"{name_prefix}.{i}",
                "load_from": load_from,
                "log_to": TrainingConfig.checkpoints_dir / f"{name_prefix}.csv",
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


def train_model(name: str, load_from: Path, log_to: Path, device: torch.device, loss_coefficients: tuple[float, ...]):
    """
    Train a model with specific loss coefficients and log results.
    """
    # Load configuration
    config = create_config(name, loss_coefficients, device=device)

    # Train model
    trainer = ConcurrentTrainer(config, load_from=load_from)
    trainer.train()

    # Log results
    for layer, coefficient, l0, ce_loss_increase in zip(
        [layer_name for layer_name in trainer.model.saes.keys()],
        loss_coefficients,
        trainer.checkpoint_l0s,
        trainer.checkpoint_ce_loss_increases,
    ):
        with log_to.open("a") as f:
            f.write(f"{layer},{coefficient:.4f},{l0:.4f},{ce_loss_increase:.4f}\n")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    match args.step:
        case 0:
            """
            Train a GPT model
            """
            # Load configuration
            config = gpt_training_options["shakespeare_64x4"]
            config.name = "experiments/regularization/shakespeare_64x4.normal"

            # Initialize trainer
            trainer = GPTTrainer(config)
            trainer.train()

        case 1:
            """
            Train a GPT model using SAE regularization
            """
            # Load configuration
            config = create_config(
                name="experiments/regularization/shakespeare_64x4.regularized",
                loss_coefficients=setup.regularization_loss_coefficients,
                max_steps=setup.regularization_max_steps,
            )

            # Initialize trainer
            trainer = RegularizationTrainer(config, torch.tensor(setup.regularization_coefficient))
            trainer.train()

        case 2:
            """
            Train SAE layers for a GPT model
            """
            # Sweep loss coefficients
            parameter_sets = []
            num_sweeps = setup.num_normal_sweeps
            for i in range(num_sweeps):
                print(f"Starting parameter sweep {i+1}/{num_sweeps}")
                sweep_training_parameters(
                    name_prefix="experiments/regularization/sae.shakespeare_64x4.normal",
                    load_from=TrainingConfig.checkpoints_dir / "experiments/regularization/shakespeare_64x4.normal",
                    starting_from=setup.sweep_normal_starting_coefficients,
                    ending_with=setup.sweep_normal_ending_coefficients,
                    steps=setup.num_normal_steps,
                )

        case 3:
            """
            Train SAE layers for a GPT model created using SAE regularization
            """
            # Sweep loss coefficients
            parameter_sets = []
            num_sweeps = setup.num_regularized_sweeps
            for i in range(num_sweeps):
                print(f"Starting parameter sweep {i+1}/{num_sweeps}")
                sweep_training_parameters(
                    name_prefix="experiments/regularization/sae.shakespeare_64x4.regularized",
                    load_from=TrainingConfig.checkpoints_dir
                    / "experiments/regularization/shakespeare_64x4.regularized",
                    starting_from=setup.sweep_regularized_starting_coefficients,
                    ending_with=setup.sweep_regularized_ending_coefficients,
                    steps=setup.num_regularized_steps,
                )

        case 4:
            """
            Process CSV files and export results.json
            """
            base_dir = TrainingConfig.checkpoints_dir / "experiments/regularization"
            column_names = ["layer", "coefficient", "l0", "ce_loss_increase"]
            normal_csv = pd.read_csv(
                base_dir / "sae.shakespeare_64x4.normal.csv",
                header=None,
                names=column_names,
            )
            regularized_csv = pd.read_csv(
                base_dir / "sae.shakespeare_64x4.regularized.csv",
                header=None,
                names=column_names,
            )

            data = {"original": [], "regularized": []}

            for layer in range(5):
                normal_data = normal_csv[normal_csv["layer"] == layer]
                regularized_data = regularized_csv[regularized_csv["layer"] == layer]

                # Create dictionaries for easy access
                normal_coefs_to_loss_increases = defaultdict(list)
                regularized_coefs_to_loss_increases = defaultdict(list)
                for row in normal_data.itertuples():
                    normal_coefs_to_loss_increases[row.coefficient].append(row)
                for row in regularized_data.itertuples():
                    regularized_coefs_to_loss_increases[row.coefficient].append(row)

                # Sort data by coefficient
                sorted_normal_data = sorted(normal_coefs_to_loss_increases.items(), key=lambda x: x[0])
                sorted_regularized_data = sorted(regularized_coefs_to_loss_increases.items(), key=lambda x: x[0])

                # Add data to dictionary
                data["original"].append(
                    [
                        {
                            "x": mean([row.l0 for row in row_set]),
                            "y": [row.ce_loss_increase for row in row_set],
                        }
                        for _, row_set in sorted_normal_data
                    ]
                )
                data["regularized"].append(
                    [
                        {
                            "x": mean([row.l0 for row in row_set]),
                            "y": [row.ce_loss_increase for row in row_set],
                        }
                        for _, row_set in sorted_regularized_data
                    ]
                )

            with open(base_dir / f"{setup.experiment_name}.results.json", "w") as f:
                json.dump(data, f, indent=4)
