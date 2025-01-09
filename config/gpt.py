from dataclasses import dataclass

from config import ConfigBase, TrainingConfigBase
from data.tokenizers import ASCIITokenizer, BaseTokenizer, TikTokenTokenizer


@dataclass
class GPTConfig(ConfigBase):
    block_size: int = 0  # max sequence length
    vocab_size: int = 0  # number of tokens
    n_layer: int = 0  # number of layers
    n_head: int = 0  # number of heads
    n_embd: int = 0  # embedding dimension

    def tokenizer(self) -> BaseTokenizer:
        """
        Infer tokenizer from vocabulary size.
        """
        match self.vocab_size:
            case TikTokenTokenizer.vocab_size:
                return TikTokenTokenizer()
            case ASCIITokenizer.vocab_size:
                return ASCIITokenizer()
            case _:
                raise ValueError(f"Unrecognized vocab size: {self.vocab_size}")


# GPT configuration options
gpt_options: dict[str, GPTConfig] = {
    "ascii_64x4": GPTConfig(
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=64,
    ),
    "ascii_128x6": GPTConfig(
        block_size=128,
        vocab_size=ASCIITokenizer.vocab_size,
        n_layer=6,
        n_head=4,
        n_embd=128,
    ),
    "tiktoken_32x4": GPTConfig(
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=32,
    ),
    "tiktoken_64x2": GPTConfig(
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=2,
        n_head=16,
        n_embd=64,
    ),
}


@dataclass
class GPTTrainingConfig(TrainingConfigBase):
    gpt_config_name: str = ""

    @property
    def gpt_config(self) -> GPTConfig:
        """
        Maps the GPT configuration name to the actual configuration.
        """
        return gpt_options[self.gpt_config_name]


# Training configuration options
training_options: dict[str, GPTTrainingConfig] = {
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
