from typing import Protocol

import torch

from config.sae.models import SAEVariant


class Experiment(Protocol):
    experiment_name: str
    sae_variant: SAEVariant
    n_features: tuple[int, ...]

    # Sweep parameters
    num_sweeps: int
    num_sweep_steps: int

    # Regularization parameters
    regularization_max_steps: int
    regularization_learning_rate: float
    regularization_min_lr: float
    regularization_coefficient: torch.Tensor
    regularization_l1_coefficients: tuple[float, ...]
    regularization_trainable_layers: tuple[int, ...] | None

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients: tuple[float, ...]
    sweep_normal_ending_coefficients: tuple[float, ...]

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients: tuple[float, ...]
    sweep_regularized_ending_coefficients: tuple[float, ...]
    sweep_regularized_ending_coefficients: tuple[float, ...]
    sweep_regularized_ending_coefficients: tuple[float, ...]


class RegularizeAllLayersExperiment(Experiment):
    """
    Use all layers when regularizing.
    """

    experiment_name = "all-layers"
    sae_variant = SAEVariant.STANDARD
    n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Sweep parameters
    num_sweeps = 1
    num_sweep_steps = 4

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-5
    regularization_coefficient = torch.tensor(3.0)
    regularization_l1_coefficients = (0.020, 0.035, 0.085, 0.07, 0.075)  # Targets l0s ~ 10
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.0021, 0.015, 0.052, 0.09, 0.26)
    sweep_normal_ending_coefficients = (0.009, 0.09, 0.33, 0.39, 0.86)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.0005, 0.002, 0.006, 0.011, 0.02)
    sweep_regularized_ending_coefficients = (0.0055, 0.014, 0.044, 0.048, 0.06)


class GatedSAEExperiment(Experiment):
    """
    Regularize using Gated SAE.
    """

    experiment_name = "gated"
    sae_variant = SAEVariant.GATED
    n_features = tuple(64 * n for n in (4, 32, 64, 64, 64))

    # Sweep parameters
    num_sweeps = 1
    num_sweep_steps = 4

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-5
    regularization_coefficient = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
    regularization_l1_coefficients = (0.1, 0.06, 0.1, 0.12, 0.2)  # Targets l0s ~ 10
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.01, 0.02, 0.05, 0.1, 0.2)
    sweep_normal_ending_coefficients = (0.08, 0.08, 0.2, 0.4, 0.8)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.005, 0.005, 0.005, 0.01, 0.02)
    sweep_regularized_ending_coefficients = (0.02, 0.02, 0.03, 0.05, 0.08)
