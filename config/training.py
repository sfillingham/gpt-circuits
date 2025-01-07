from dataclasses import dataclass
from typing import Optional

import torch

from config.gpt import GPTConfig
from config.gpt import options as gpt_options


@dataclass
class TrainingConfig:
    # Path parameters
    out_dir: str = ""
    data_dir: str = ""
    should_randomize: bool = True

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
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0  # Maximum gradient norm

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
    "shakespeare_64x4": TrainingConfig(
        out_dir="checkpoints/shakespeare_64x4",
        data_dir="data/shakespeare",
        gpt_config_name="ascii_64x4",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=1000,
        max_steps=9000,
        decay_lr=True,
        lr_decay_steps=9000,
        min_lr=1e-4,
    ),
    "shakespeare_128x6": TrainingConfig(
        out_dir="checkpoints/shakespeare_128x6",
        data_dir="data/shakespeare",
        gpt_config_name="ascii_128x6",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=300,
        max_steps=3000,
        decay_lr=True,
        lr_decay_steps=3000,
        min_lr=1e-4,
    ),
    "tiny_32x4": TrainingConfig(
        out_dir="checkpoints/tiny_32x4",
        data_dir="data/tiny_stories_10m",
        gpt_config_name="tiktoken_32x4",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
    "tiny_64x2": TrainingConfig(
        out_dir="checkpoints/tiny_64x2",
        data_dir="data/tiny_stories_10m",
        gpt_config_name="tiktoken_64x2",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
}
