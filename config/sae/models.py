from dataclasses import dataclass
from enum import Enum

from config import ConfigBase
from config.gpt.models import GPTConfig, gpt_options


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
        return gpt_options[self.gpt_config_name]


# SAE configuration options
sae_options: dict[str, SAEConfig] = {
    "gated_v2_shakespeare_64x4": SAEConfig(
        gpt_config_name="ascii_64x4",
        n_features=tuple(64 * n for n in (4, 4, 4, 8, 16)),
        sae_variant=SAEVariant.GATED_V2,
    ),
}
