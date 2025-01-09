from dataclasses import dataclass
from typing import Optional, TypeVar

import torch


@dataclass
class ConfigBase:
    name: str

    @property
    def device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @property
    def compile(self):
        """
        Can only compile on CUDA
        """
        return self.device == "cuda"


@dataclass
class TrainingConfigBase(ConfigBase):
    # Data parameters
    data_dir: str = ""
    should_randomize: bool = True

    # Evaluation parameters
    eval_interval: int = 0
    eval_steps: int = 0

    # Batch parameters
    batch_size: int = 0
    gradient_accumulation_steps: int = 0

    # Training parameters
    learning_rate: float = 0
    warmup_steps: int = 0
    max_steps: int = 0
    decay_lr: bool = False
    min_lr: float = 0  # Minimum learning rate
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0  # Maximum gradient norm

    @property
    def out_dir(self) -> str:
        """
        Checkpoint path
        """
        return f"checkpoints/{self.name}"


Config = TypeVar("Config", bound=ConfigBase)


def map_options(*options: Config) -> dict[str, Config]:
    """
    Map configurations to a dictionary using name as key.
    """
    return {option.name: option for option in options}
