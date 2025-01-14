from dataclasses import dataclass, field
from enum import Enum

from config import Config, map_options
from config.gpt.models import GPTConfig, gpt_options


class SAEVariant(str, Enum):
    GATED = "gated"
    GATED_V2 = "gated_v2"


@dataclass
class SAEConfig(Config):
    gpt_config: GPTConfig = field(default_factory=GPTConfig)
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.GATED_V2

    @property
    def block_size(self) -> int:
        return self.gpt_config.block_size


# SAE configuration options
sae_options: dict[str, SAEConfig] = map_options(
    SAEConfig(
        name="gated_v2x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.GATED_V2,
    ),
    SAEConfig(
        name="gated_v2x32.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (32, 32, 32, 32, 32)),
        sae_variant=SAEVariant.GATED_V2,
    ),
)
