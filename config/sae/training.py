from dataclasses import dataclass, field
from typing import Optional

from config import TrainingConfigBase

from .models import SAEConfig, sae_options


@dataclass
class LossCoefficients:
    l1: tuple = ()


@dataclass
class SAETrainingConfig(TrainingConfigBase):
    sae_config_name: str = ""
    trainable_layers: Optional[tuple] = None  # If none, all layers are trained.
    loss_coefficients: LossCoefficients = field(default_factory=LossCoefficients)

    @property
    def sae_config(self) -> SAEConfig:
        return sae_options[self.sae_config_name]


# Training configuration options
options: dict[str, SAETrainingConfig] = {
    "gated_v2_shakespeare_64x4": SAETrainingConfig(
        sae_config_name="gated_v2_shakespeare_64x4",
        out_dir="checkpoints/gated_v2_shakespeare_64x4",
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=750,
        max_steps=7500,
        decay_lr=True,
        min_lr=1e-4,
        loss_coefficients=LossCoefficients(
            l1=(0.5, 1.5, 1.5, 4.0, 9.0),
        ),
    ),
}
