from dataclasses import dataclass

from config import TrainingConfigBase, map_options

from .models import GPTConfig, gpt_options


@dataclass
class GPTTrainingConfig(TrainingConfigBase):
    gpt_config_name: str = ""

    @property
    def gpt_config(self) -> GPTConfig:
        return gpt_options[self.gpt_config_name]


# Training configuration options
options: dict[str, GPTTrainingConfig] = map_options(
    GPTTrainingConfig(
        name="shakespeare_64x4",
        gpt_config_name="ascii_64x4",
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
    GPTTrainingConfig(
        name="shakespeare_128x6",
        gpt_config_name="ascii_128x6",
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
    GPTTrainingConfig(
        name="tiny_32x4",
        gpt_config_name="tiktoken_32x4",
        data_dir="data/tiny_stories_10m",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
    GPTTrainingConfig(
        name="tiny_64x2",
        gpt_config_name="tiktoken_64x2",
        data_dir="data/tiny_stories_10m",
        eval_interval=100,
        eval_steps=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        learning_rate=1e-3,
        max_steps=5000,
    ),
)
