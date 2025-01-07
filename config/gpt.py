from dataclasses import dataclass

from data.tokenizers import ASCIITokenizer, BaseTokenizer, TikTokenTokenizer


@dataclass
class GPTConfig:
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


# Configuration options
options: dict[str, GPTConfig] = {
    "32x4": GPTConfig(
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=4,
        n_head=16,
        n_embd=32,
    ),
    "64x2": GPTConfig(
        block_size=128,
        vocab_size=TikTokenTokenizer.vocab_size,
        n_layer=2,
        n_head=16,
        n_embd=64,
    ),
}
