from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import ConfigBase, TrainingConfigBase
from config.gpt import GPTConfig, gpt_options


class SAEVariant(str, Enum):
    GATED = "gated"
    GATED_V2 = "gated_v2"
    JUMP_RELU = "jumprelu"


@dataclass
class SAEConfig(ConfigBase):
    gpt_config_name: str = ""
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.GATED_V2

    @property
    def gpt_config(self) -> GPTConfig:
        """
        Maps the GPT configuration name to the actual configuration.
        """
        return gpt_options[self.gpt_config_name]


# SAE configuration options
sae_options: dict[str, SAEConfig] = {
    "gated_v2_shakespeare_64x4": SAEConfig(
        gpt_config_name="ascii_64x4",
        n_features=tuple(64 * n for n in (4, 4, 4, 8, 16)),
        sae_variant=SAEVariant.GATED_V2,
    ),
}


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
        """
        Maps the SAE configuration name to the actual configuration.
        """
        return sae_options[self.sae_config_name]


# Training configuration options
sae_training_options: dict[str, SAETrainingConfig] = {
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
