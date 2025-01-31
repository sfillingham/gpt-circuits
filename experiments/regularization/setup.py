from typing import Protocol

import torch

from config.sae.models import SAEVariant
from config.sae.training import LossCoefficients


class Experiment(Protocol):
    experiment_name: str
    sae_variant: SAEVariant
    n_features: tuple[int, ...]

    # Regularization parameters
    regularization_max_steps: int
    regularization_learning_rate: float
    regularization_min_lr: float
    regularization_loss_coefficients: LossCoefficients

    # Sweep parameters
    num_sweep_steps: int

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients: tuple[float, ...]
    sweep_normal_ending_coefficients: tuple[float, ...]

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients: tuple[float, ...]
    sweep_regularized_ending_coefficients: tuple[float, ...]


class RegularizeAllLayersExperiment(Experiment):
    """
    Use all layers when regularizing.
    """

    experiment_name = "all-layers"
    sae_variant = SAEVariant.STANDARD
    n_features = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-5
    regularization_loss_coefficients = LossCoefficients(
        sparsity=(0.020, 0.035, 0.085, 0.07, 0.075),  # Targets l0s ~ 10
        regularization=torch.tensor(3.0),
    )

    # Sweep parameters
    num_sweep_steps = 8

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.0021, 0.0175, 0.056, 0.09, 0.275)
    sweep_normal_ending_coefficients = (0.009, 0.105, 0.33, 0.39, 0.87)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.0005, 0.002, 0.006, 0.010, 0.02)
    sweep_regularized_ending_coefficients = (0.0055, 0.014, 0.044, 0.048, 0.06)


class GatedExperiment(Experiment):
    """
    Regularize using Gated SAE.
    """

    experiment_name = "gated"
    sae_variant = SAEVariant.GATED
    n_features = tuple(64 * n for n in (4, 32, 64, 64, 64))

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-5
    regularization_loss_coefficients = LossCoefficients(
        sparsity=(0.100, 0.035, 0.075, 0.080, 0.080),  # Targets l0s ~ 10
        regularization=torch.tensor(2.0),
    )

    # Sweep parameters
    num_sweep_steps = 4

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.01, 0.02, 0.05, 0.1, 0.2)
    sweep_normal_ending_coefficients = (0.08, 0.10, 0.25, 0.4, 0.8)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.002, 0.002, 0.005, 0.015, 0.02)
    sweep_regularized_ending_coefficients = (0.04, 0.024, 0.04, 0.07, 0.08)


class JumpReLUExperiment(Experiment):
    """
    Regularize using standard SAE and sweep using JumpReLU.
    """

    experiment_name = "jumprelu"
    sae_variant = SAEVariant.JUMP_RELU
    n_features = tuple(64 * n for n in (4, 32, 64, 64, 64))

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-5
    regularization_loss_coefficients = LossCoefficients(
        sparsity=(0.01, 0.01, 0.01, 0.03, 0.08),  # Targets l0s ~ 10
        regularization=torch.tensor(3.0),
        bandwidth=0.1,
    )

    # Sweep parameters
    num_sweep_steps = 4

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.001, 0.001, 0.001, 0.001, 0.001)
    sweep_normal_ending_coefficients = (0.1, 0.1, 0.1, 0.1, 0.1)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.001, 0.001, 0.001, 0.001, 0.001)
    sweep_regularized_ending_coefficients = (0.1, 0.1, 0.1, 0.1, 0.1)
