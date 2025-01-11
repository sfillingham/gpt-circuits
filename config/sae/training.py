from dataclasses import dataclass, field
from typing import Optional

from config import TrainingConfig, map_options

from .models import SAEConfig, sae_options


@dataclass
class LossCoefficients:
    l1: tuple = ()


@dataclass
class SAETrainingConfig(TrainingConfig):
    sae_config_name: str = ""
    trainable_layers: Optional[tuple] = None  # If none, all layers are trained.
    loss_coefficients: LossCoefficients = field(default_factory=LossCoefficients)

    @property
    def sae_config(self) -> SAEConfig:
        return sae_options[self.sae_config_name]


# Shared training parameters
gated_v2x8_shakespeare_64x4_defaults = {
    "sae_config_name": "gated_v2x8_shakespeare_64x4",
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
        name="training.a.gated_v2x8_shakespeare_64x4",
        **gated_v2x8_shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=(0.5, 0.5, 1.0, 1.5, 2.0),
        ),
    ),
    SAETrainingConfig(
        name="training.b.gated_v2x8_shakespeare_64x4",
        **gated_v2x8_shakespeare_64x4_defaults,
        loss_coefficients=LossCoefficients(
            l1=(0.5, 0.5, 1.5, 2.0, 3.0),
        ),
    ),
)
