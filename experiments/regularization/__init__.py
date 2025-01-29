import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from config.gpt.models import GPTConfig, gpt_options
from config.sae.models import SAEConfig
from config.sae.training import (
    LossCoefficients,
    SAETrainingConfig,
    shakespeare_64x4_defaults,
)
from experiments import ParameterSweeper
from experiments.regularization.setup import (  # noqa: F401
    Experiment,
    RegularizeAllLayersExperiment,
)
from training.sae.concurrent import ConcurrentTrainer


def create_config(
    setup: Experiment,
    name: str,
    sparsity_coefficients: tuple[float, ...],
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
            sparsity=sparsity_coefficients,
        ),
        trainable_layers=trainable_layers,
    )

    # Set optional args
    config.device = device or config.device

    return config


def sweep_training_parameters(
    setup: Experiment,
    name_prefix: str,
    log_layers_to: Path,
    log_sums_to: Path,
    load_from: Path,
    starting_from: tuple[float, ...],
    ending_with: tuple[float, ...],
    steps: int,
):
    """
    Sweep over a range of parameters.
    """
    parameter_sets = []
    for i in range(steps):
        coefficients = tuple(start + (end - start) * i / (steps - 1) for start, end in zip(starting_from, ending_with))
        parameter_sets.append(
            {
                "setup": setup,
                "name": f"{name_prefix}.{i}",
                "load_from": load_from,
                "log_layers_to": log_layers_to,
                "log_sums_to": log_sums_to,
                "sparsity_coefficients": coefficients,
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
    setup: Experiment,
    name: str,
    load_from: Path,
    log_layers_to: Path,
    log_sums_to: Path,
    device: torch.device,
    sparsity_coefficients: tuple[float, ...],
):
    """
    Train a model with specific loss coefficients and log results.
    """
    # Load configuration
    config = create_config(setup, name, sparsity_coefficients, device=device)

    # Train model
    trainer = ConcurrentTrainer(config, load_from=load_from)
    trainer.train()

    # Log layers
    for layer, coefficient, l0, ce_loss_increase in zip(
        [layer_name for layer_name in trainer.model.saes.keys()],
        sparsity_coefficients,
        trainer.checkpoint_l0s,
        trainer.checkpoint_ce_loss_increases,
    ):
        with log_layers_to.open("a") as f:
            f.write(f"{layer},{coefficient:.6f},{l0:.6f},{ce_loss_increase:.6f}\n")

    # Log end-to-end metrics
    with log_sums_to.open("a") as f:
        data = []
        data.append(round(sum(sparsity_coefficients), 6))
        data.append(round(sum(trainer.checkpoint_l0s.tolist()), 6))
        data.append(round(trainer.checkpoint_compound_ce_loss_increase.item(), 6))
        f.write(",".join(map(str, data)) + "\n")


def export_sweep_results(setup: Experiment, base_dir: Path):
    """
    Export sweep results to JSON files.
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

    with open(base_dir / f"{setup.experiment_name}.results.sweep.json", "w") as f:
        json.dump(data, f, indent=4)


def export_sums(setup: Experiment, base_dir: Path):
    """
    Export sums and compound losses to JSON files.
    """
    # Read CE losses
    normal_ce_losses = pd.read_csv(base_dir / "model.normal.csv", header=None, names=("ce_loss",))
    regularized_ce_losses = pd.read_csv(
        base_dir / f"{setup.experiment_name}.model.regularized.csv", header=None, names=("ce_loss",)
    )

    # CE loss data
    ce_loss_data = {"original": [], "regularized": []}
    for ce_loss in normal_ce_losses.itertuples():
        ce_loss_data["original"].append({"ce_loss": ce_loss.ce_loss})
    for ce_loss in regularized_ce_losses.itertuples():
        ce_loss_data["regularized"].append({"ce_loss": ce_loss.ce_loss})

    # Export to JSON
    with open(base_dir / f"{setup.experiment_name}.results.models.json", "w") as f:
        json.dump(ce_loss_data, f, indent=4)

    # Read sums
    column_names = ["sum_coeffs", "sum_l0s", "ce_loss_increase"]
    normal_csv = pd.read_csv(
        base_dir / "sums.normal.csv",
        header=None,
        names=column_names,
    )
    regularized_csv = pd.read_csv(
        base_dir / f"{setup.experiment_name}.sums.regularized.csv",
        header=None,
        names=column_names,
    )

    # Group by 'sum_coeffs' and calculate the mean for 'sum_l0s' and 'ce_loss_increase'
    grouped_normal = normal_csv.groupby("sum_coeffs")[["sum_l0s", "ce_loss_increase"]].mean().reset_index()
    grouped_regularized = regularized_csv.groupby("sum_coeffs")[["sum_l0s", "ce_loss_increase"]].mean().reset_index()

    # Sweep data
    sweep_data = {"original": [], "regularized": []}
    for _, row in grouped_normal.iterrows():
        sweep_data["original"].append(
            {
                "sum_coeffs": round(row["sum_coeffs"], 6),
                "sum_l0s": round(row["sum_l0s"], 6),
                "ce_loss_increase": round(row["ce_loss_increase"], 6),
            }
        )
    for _, row in grouped_regularized.iterrows():
        sweep_data["regularized"].append(
            {
                "sum_coeffs": round(row["sum_coeffs"], 6),
                "sum_l0s": round(row["sum_l0s"], 6),
                "ce_loss_increase": round(row["ce_loss_increase"], 6),
            }
        )

    # Export to JSON
    with open(base_dir / f"{setup.experiment_name}.results.sums.json", "w") as f:
        json.dump(sweep_data, f, indent=4)
