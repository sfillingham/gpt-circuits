from dataclasses import dataclass

from config import TrainingConfigBase

from .models import GPTConfig, gpt_options


@dataclass
class GPTTrainingConfig(TrainingConfigBase):
    gpt_config_name: str = ""

    @property
    def gpt_config(self) -> GPTConfig:
        return gpt_options[self.gpt_config_name]


# Training configuration options
options: dict[str, GPTTrainingConfig] = {
    "shakespeare_64x4": GPTTrainingConfig(
        gpt_config_name="ascii_64x4",
        out_dir="checkpoints/shakespeare_64x4",
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
    ),
    "shakespeare_128x6": GPTTrainingConfig(
        gpt_config_name="ascii_128x6",
        out_dir="checkpoints/shakespeare_128x6",
        data_dir="data/shakespeare",
        eval_interval=250,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=300,
        max_steps=3000,
        decay_lr=True,
        min_lr=1e-4,
    ),
    "tiny_32x4": GPTTrainingConfig(
        gpt_config_name="tiktoken_32x4",
        out_dir="checkpoints/tiny_32x4",
        data_dir="data/tiny_stories_10m",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
    "tiny_64x2": GPTTrainingConfig(
        gpt_config_name="tiktoken_64x2",
        out_dir="checkpoints/tiny_64x2",
        data_dir="data/tiny_stories_10m",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
}
