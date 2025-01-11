from dataclasses import dataclass
from enum import Enum

from config import Config, map_options
from config.gpt.models import GPTConfig, gpt_options


class SAEVariant(str, Enum):
    GATED = "gated"
    GATED_V2 = "gated_v2"


@dataclass
class SAEConfig(Config):
    gpt_config_name: str = ""
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.GATED_V2

    @property
    def gpt_config(self) -> GPTConfig:
        return gpt_options[self.gpt_config_name]

    @property
    def block_size(self) -> int:
        return self.gpt_config.block_size


# SAE configuration options
sae_options: dict[str, SAEConfig] = map_options(
    SAEConfig(
        name="gated_v2_shakespeare_64x4",
        gpt_config_name="ascii_64x4",
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.GATED_V2,
    ),
)
