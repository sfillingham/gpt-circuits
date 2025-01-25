"""
Train a GPT model with experimental SAE regularization.

$ python -m experiments.regularization.train --setup=all-layers --step=0
"""

import argparse

from config import TrainingConfig
from config.gpt.training import options as gpt_training_options
from experiments.regularization import (
    create_config,
    export_e2e_results,
    export_sweep_results,
    sweep_training_parameters,
)
from experiments.regularization.setup import (  # noqa: F401
    GatedSAEExperiment,
    RegularizeAllLayersExperiment,
)
from training.gpt import GPTTrainer
from training.sae.regularization import RegularizationTrainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Which step to run")
    parser.add_argument("--setup", type=str, default="all-layers", help="Which setup to use")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Experiment setups are in setup.py
    match args.setup:
        case "all-layers":
            setup = RegularizeAllLayersExperiment()
        case "gated":
            setup = GatedSAEExperiment()

    base_dir = TrainingConfig.checkpoints_dir / "regularization"

    match args.step:
        case 0:
            """
            Train a GPT model
            """
            # Load configuration
            config = gpt_training_options["shakespeare_64x4"]
            config.name = "regularization/model.normal"
            # Train using same params as regularized model
            config.max_steps = setup.regularization_max_steps
            config.learning_rate = setup.regularization_learning_rate
            config.min_lr = setup.regularization_min_lr

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
                setup=setup,
                name=f"regularization/{setup.experiment_name}.model.regularized",
                l1_coefficients=setup.regularization_l1_coefficients,
                trainable_layers=setup.regularization_trainable_layers,
            )
            # Set learning rate parameters
            config.max_steps = setup.regularization_max_steps
            config.learning_rate = setup.regularization_learning_rate
            config.min_lr = setup.regularization_min_lr
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
            for i in range(setup.num_sweeps):
                print(f"Starting parameter sweep {i+1}/{setup.num_sweeps}")
                sweep_training_parameters(
                    setup=setup,
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
            for i in range(setup.num_sweeps):
                print(f"Starting parameter sweep {i+1}/{setup.num_sweeps}")
                sweep_training_parameters(
                    setup=setup,
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
            export_sweep_results(setup, base_dir)
            export_e2e_results(setup, base_dir)
