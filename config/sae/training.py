from dataclasses import dataclass, field
from typing import Optional

import torch

from config import TrainingConfig, map_options

from .models import SAEConfig, sae_options


@dataclass
class LossCoefficients:
    l1: tuple[float, ...] = ()
    regularization: Optional[torch.Tensor] = None  # For experiments


@dataclass
class SAETrainingConfig(TrainingConfig):
    sae_config: SAEConfig = field(default_factory=SAEConfig)
    trainable_layers: Optional[tuple] = None  # If none, all layers are trained.
    loss_coefficients: LossCoefficients = field(default_factory=LossCoefficients)


# Shared training parameters
shakespeare_64x4_defaults = {
    "data_dir": "data/shakespeare",
    "eval_interval": 250,
    "eval_steps": 100,
    "batch_size": 128,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "warmup_steps": 750,
    "max_steps": 7500,
    "decay_lr": True,
    "min_lr": 1e-4,
}


# Training configuration options
options: dict[str, SAETrainingConfig] = map_options(
    SAETrainingConfig(
        name="standardx8.shakespeare_64x4.v0",
        sae_config=sae_options["standardx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=(0.001, 0.001, 0.002, 0.003, 0.006),
        ),
    ),
    SAETrainingConfig(
        name="standardx8.shakespeare_64x4.v1",
        sae_config=sae_options["standardx8.shakespeare_64x4"],
        **shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=(0.01, 0.03, 0.05, 0.05, 0.05),
            regularization=torch.tensor(10000.0),
        ),
    ),
)
