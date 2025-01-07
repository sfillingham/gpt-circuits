from dataclasses import dataclass

import torch

from config.gpt import GPTConfig
from config.gpt import options as gpt_options


@dataclass
class TrainingConfig:
    # Path parameters
    out_dir: str = ""
    data_dir: str = ""
    should_randomize: bool = False

    # Model configuration
    gpt_config_name: str = ""

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
    lr_decay_steps: int = 0  # For how many iterations should the learning rate decay?
    min_lr: float = 0  # Minimum learning rate
    weight_decay: float = 0
    grad_clip: float = 0  # Maximum gradient norm

    @property
    def gpt_config(self) -> GPTConfig:
        """
        Maps the GPT configuration name to the actual configuration.
        """
        return gpt_options[self.gpt_config_name]

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


# Configuration options
options: dict[str, TrainingConfig] = {
    "32x4": TrainingConfig(
        out_dir="checkpoints/32x4",
        data_dir="data/pile_10k",
        should_randomize=True,
        gpt_config_name="32x4",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=5000,
        decay_lr=False,
        lr_decay_steps=0,
        min_lr=0,
        weight_decay=0.1,
        grad_clip=1.0,
    ),
    "64x2": TrainingConfig(
        out_dir="checkpoints/64x2",
        data_dir="data/pile_10k",
        should_randomize=True,
        gpt_config_name="64x2",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=5000,
        decay_lr=False,
        lr_decay_steps=0,
        min_lr=0,
        weight_decay=0.1,
        grad_clip=1.0,
    ),
}
