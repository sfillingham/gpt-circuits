"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.shakespeare_64x4 --experiment=0
"""

import argparse
import json
from pathlib import Path

import pandas as pd
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
from config.sae.training import options as sae_training_options
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

    match args.experiment:
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
            config = sae_training_options["train.a.gated_v2x8.shakespeare_64x4"]
            config.name = "experiments/regularization/shakespeare_64x4.regularized"
            config.sae_config.n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))
            config.loss_coefficients.l1 = (0.5, 0.5, 1.0, 3.5, 3.5)
            config.max_steps = 15000

            # Initialize trainer
            trainer = RegularizationTrainer(config, torch.tensor(100.0))
            trainer.train()

        case 2:
            """
            Train SAE layers for a GPT model
            """
            # Sweep loss coefficients
            # The following coefficients yield l0s ~ 10.0: (0.4, 0.8, 7.5, 23.0, 40.0)
            parameter_sets = []
            starting_coefficients = (0.1, 0.2, 1.0, 8.0, 14.0)
            ending_coefficients = (2.0, 4.0, 20.0, 65.0, 90.0)
            num_sweeps = 5
            for i in range(num_sweeps):
                print(f"Starting parameter sweep {i}/{num_sweeps}")
                sweep_training_parameters(
                    name_prefix="experiments/regularization/sae.shakespeare_64x4.normal",
                    load_from=TrainingConfig.checkpoints_dir / "experiments/regularization/shakespeare_64x4.normal",
                    starting_from=starting_coefficients,
                    ending_with=ending_coefficients,
                    steps=24,
                )

        case 3:
            """
            Train SAE layers for a GPT model created using SAE regularization
            """
            # Sweep loss coefficients
            # The following coefficients yield l0s ~ 10.0: (0.15, 0.33, 1.5, 3.8, 4.1)
            parameter_sets = []
            starting_coefficients = (0.04, 0.1, 0.3, 0.5, 1.0)
            ending_coefficients = (0.80, 2.0, 6.0, 10.0, 20.0)
            num_sweeps = 5
            for i in range(num_sweeps):
                print(f"Starting parameter sweep {i}/{num_sweeps}")
                sweep_training_parameters(
                    name_prefix="experiments/regularization/sae.shakespeare_64x4.regularized",
                    load_from=TrainingConfig.checkpoints_dir
                    / "experiments/regularization/shakespeare_64x4.regularized",
                    starting_from=starting_coefficients,
                    ending_with=ending_coefficients,
                    steps=24,
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

                # Group and average data by coefficient
                normal_data = normal_data.groupby("coefficient").mean().reset_index()
                regularized_data = regularized_data.groupby("coefficient").mean().reset_index()

                # Sort data by L0
                normal_data = normal_data.sort_values("l0")
                regularized_data = regularized_data.sort_values("l0")

                # Add data to dictionary
                data["original"].append(
                    [
                        {
                            "x": row.l0,
                            "y": row.ce_loss_increase,
                        }
                        for row in normal_data.itertuples()
                    ]
                )
                data["regularized"].append(
                    [
                        {
                            "x": row.l0,
                            "y": row.ce_loss_increase,
                        }
                        for row in regularized_data.itertuples()
                    ]
                )

            with open(base_dir / "shakespeare_64x4.results.json", "w") as f:
                json.dump(data, f, indent=4)
