"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.train --step=0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

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
    LayersExperimentSetup,
    StandardExperimentSetup,
)
from training.gpt import GPTTrainer
from training.sae.concurrent import ConcurrentTrainer
from training.sae.regularization import RegularizationTrainer

# Experiment setups are in setup.py
setup = LayersExperimentSetup()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Which step to run")
    return parser.parse_args()


def create_config(
    name: str,
    l1_coefficients: tuple[float, ...],
    trainable_layers: tuple[int, ...] | None = None,
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
            l1=l1_coefficients,
        ),
        trainable_layers=trainable_layers,
    )

    # Set optional args
    config.device = device or config.device
    config.max_steps = max_steps or config.max_steps

    return config


def sweep_training_parameters(
    name_prefix: str,
    log_to: Path,
    load_from: Path,
    starting_from: tuple[float, ...],
    ending_with: tuple[float, ...],
    steps: int,
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
                "log_to": log_to,
                "l1_coefficients": coefficients,
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


def train_model(name: str, load_from: Path, log_to: Path, device: torch.device, l1_coefficients: tuple[float, ...]):
    """
    Train a model with specific loss coefficients and log results.
    """
    # Load configuration
    config = create_config(name, l1_coefficients, device=device)

    # Train model
    trainer = ConcurrentTrainer(config, load_from=load_from)
    trainer.train()

    # Log results
    for layer, coefficient, l0, ce_loss_increase in zip(
        [layer_name for layer_name in trainer.model.saes.keys()],
        l1_coefficients,
        trainer.checkpoint_l0s,
        trainer.checkpoint_ce_loss_increases,
    ):
        with log_to.open("a") as f:
            f.write(f"{layer},{coefficient:.4f},{l0:.4f},{ce_loss_increase:.4f}\n")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    base_dir = TrainingConfig.checkpoints_dir / "regularization"

    match args.step:
        case 0:
            """
            Train a GPT model
            """
            # Load configuration
            config = gpt_training_options["shakespeare_64x4"]
            config.name = f"regularization/{setup.experiment_name}.model.normal"

            # Initialize trainer
            trainer = GPTTrainer(config)
            trainer.train()

            # Log final CE loss
            with (base_dir / f"{setup.experiment_name}.model.normal.csv").open("a") as f:
                f.write(f"{trainer.best_val_loss:.4f}\n")

        case 1:
            """
            Train a GPT model using SAE regularization
            """
            # Load configuration
            config = create_config(
                name=f"regularization/{setup.experiment_name}.model.regularized",
                l1_coefficients=setup.regularization_l1_coefficients,
                max_steps=setup.regularization_max_steps,
                trainable_layers=setup.regularization_trainable_layers,
            )

            # Initialize trainer
            trainer = RegularizationTrainer(config, torch.tensor(setup.regularization_coefficient))
            trainer.train()

            # Log final CE loss
            with (base_dir / f"{setup.experiment_name}.model.regularized.csv").open("a") as f:
                f.write(f"{trainer.checkpoint_ce_loss:.4f}\n")

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
                    name_prefix="regularization/saes/normal",
                    log_to=base_dir / f"{setup.experiment_name}.saes.normal.csv",
                    load_from=base_dir / f"{setup.experiment_name}.model.normal",
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
                    name_prefix="regularization/saes/regularized",
                    log_to=base_dir / f"{setup.experiment_name}.saes.regularized.csv",
                    load_from=base_dir / f"{setup.experiment_name}.model.regularized",
                    starting_from=setup.sweep_regularized_starting_coefficients,
                    ending_with=setup.sweep_regularized_ending_coefficients,
                    steps=setup.num_regularized_steps,
                )

        case 4:
            """
            Process CSV files and export results.json
            """
            column_names = ["layer", "coefficient", "l0", "ce_loss_increase"]
            normal_csv = pd.read_csv(
                base_dir / f"{setup.experiment_name}.saes.normal.csv",
                header=None,
                names=column_names,
            )
            regularized_csv = pd.read_csv(
                base_dir / f"{setup.experiment_name}.saes.regularized.csv",
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
                            "coefficient": coefficient,
                            "x": [row.l0 for row in row_set],
                            "y": [row.ce_loss_increase for row in row_set],
                        }
                        for coefficient, row_set in sorted_normal_data
                    ]
                )
                data["regularized"].append(
                    [
                        {
                            "coefficient": coefficient,
                            "x": [row.l0 for row in row_set],
                            "y": [row.ce_loss_increase for row in row_set],
                        }
                        for coefficient, row_set in sorted_regularized_data
                    ]
                )

            with open(base_dir / f"{setup.experiment_name}.results.json", "w") as f:
                json.dump(data, f, indent=4)
