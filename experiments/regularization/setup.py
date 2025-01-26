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
    num_sweep_steps = 8

    # Regularization parameters
    regularization_max_steps = 15000
    regularization_learning_rate = 1e-3
    regularization_min_lr = 1e-4
    regularization_coefficient = torch.tensor(10000.0)
    regularization_l1_coefficients = (0.01, 0.03, 0.05, 0.06, 0.07)  # Targets l0s ~ 10
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.00006, 0.00025, 0.0010, 0.0015, 0.0046)
    sweep_normal_ending_coefficients = (0.00016, 0.00145, 0.0048, 0.0050, 0.0140)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.000035, 0.00005, 0.00017, 0.0003, 0.00055)
    sweep_regularized_ending_coefficients = (0.00014, 0.00030, 0.00095, 0.0010, 0.00155)


class GatedSAEExperiment(Experiment):
    """
    Vary the coefficients used.
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
    regularization_coefficient = torch.tensor([10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    regularization_l1_coefficients = (10.0, 40.0, 225.0, 225.0, 225.0)  # Targets l0s ~ 10
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.1, 1.0, 4.0, 8.0, 16.0)
    sweep_normal_ending_coefficients = (0.5, 4.0, 20.0, 30.0, 60.0)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.05, 0.3, 1.0, 1.0, 2.0)
    sweep_regularized_ending_coefficients = (0.25, 1.3, 5.0, 6.0, 12.0)
