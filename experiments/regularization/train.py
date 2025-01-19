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
from experiments.regularization.setup import RegularizeAllLayersExperiment  # noqa: F401
from training.gpt import GPTTrainer
from training.sae.concurrent import ConcurrentTrainer
from training.sae.regularization import RegularizationTrainer

# Experiment setups are in setup.py
setup = RegularizeAllLayersExperiment()


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

    return config


def sweep_training_parameters(
    name_prefix: str,
    log_layers_to: Path,
    log_e2e_to: Path,
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
                "log_layers_to": log_layers_to,
                "log_e2e_to": log_e2e_to,
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


def train_model(
    name: str,
    load_from: Path,
    log_layers_to: Path,
    log_e2e_to: Path,
    device: torch.device,
    l1_coefficients: tuple[float, ...],
):
    """
    Train a model with specific loss coefficients and log results.
    """
    # Load configuration
    config = create_config(name, l1_coefficients, device=device)

    # Train model
    trainer = ConcurrentTrainer(config, load_from=load_from)
    trainer.train()

    # Log layers
    for layer, coefficient, l0, ce_loss_increase in zip(
        [layer_name for layer_name in trainer.model.saes.keys()],
        l1_coefficients,
        trainer.checkpoint_l0s,
        trainer.checkpoint_ce_loss_increases,
    ):
        with log_layers_to.open("a") as f:
            f.write(f"{layer},{coefficient:.6f},{l0:.6f},{ce_loss_increase:.6f}\n")

    # Log end-to-end metrics
    with log_e2e_to.open("a") as f:
        data = []
        data.append(round(sum(l1_coefficients), 6))
        data.append(round(sum(trainer.checkpoint_l0s.tolist()), 6))
        data.append(round(trainer.checkpoint_e2e_ce_loss_increase.item(), 6))
        data.append(round(trainer.checkpoint_e2e_kl_div.item(), 6))
        f.write(",".join(map(str, data)) + "\n")


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
            config.name = "regularization/model.normal"
            # Train for same duration
            config.max_steps = setup.regularization_max_steps

            # Initialize trainer
            trainer = GPTTrainer(config)
            trainer.train()

            # Log final CE loss
            with (base_dir / "model.normal.csv").open("a") as f:
                f.write(f"{trainer.best_val_loss:.4f}\n")

        case 1:
            """
            Train a GPT model using SAE regularization
            """
            # Load configuration
            config = create_config(
                name=f"regularization/{setup.experiment_name}.model.regularized",
                l1_coefficients=setup.regularization_l1_coefficients,
                trainable_layers=setup.regularization_trainable_layers,
            )
            config.max_steps = setup.regularization_max_steps
            config.loss_coefficients.regularization = setup.regularization_coefficient

            # Initialize trainer
            trainer = RegularizationTrainer(config)
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
            for i in range(setup.num_sweeps):
                print(f"Starting parameter sweep {i+1}/{setup.num_sweeps}")
                sweep_training_parameters(
                    name_prefix="regularization/saes/normal",
                    log_layers_to=base_dir / "saes.normal.csv",
                    log_e2e_to=base_dir / "e2e.normal.csv",
                    load_from=base_dir / "model.normal",
                    starting_from=setup.sweep_normal_starting_coefficients,
                    ending_with=setup.sweep_normal_ending_coefficients,
                    steps=setup.num_sweep_steps,
                )

        case 3:
            """
            Train SAE layers for a GPT model created using SAE regularization
            """
            # Sweep loss coefficients
            parameter_sets = []
            for i in range(setup.num_sweeps):
                print(f"Starting parameter sweep {i+1}/{setup.num_sweeps}")
                sweep_training_parameters(
                    name_prefix="regularization/saes/regularized",
                    log_layers_to=base_dir / f"{setup.experiment_name}.saes.regularized.csv",
                    log_e2e_to=base_dir / f"{setup.experiment_name}.e2e.regularized.csv",
                    load_from=base_dir / f"{setup.experiment_name}.model.regularized",
                    starting_from=setup.sweep_regularized_starting_coefficients,
                    ending_with=setup.sweep_regularized_ending_coefficients,
                    steps=setup.num_sweep_steps,
                )

        case 4:
            """
            Process CSV files and export results.json
            """
            column_names = ["layer", "coefficient", "l0", "ce_loss_increase"]
            normal_csv = pd.read_csv(
                base_dir / "saes.normal.csv",
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
