"""
This module contains classes for tokenizing text.
"""

import tiktoken


class BaseTokenizer:
    """
    Base class for tokenizers.
    """

    # The size of the vocabulary.
    vocab_size: int

    def encode(self, text) -> list[int]:
        """
        Tokenizes the input text and returns a list of token IDs.
        """
        raise NotImplementedError()

    def decode_sequence(self, tokens: list[int]) -> str:
        """
        Converts a list of token IDs to a string.
        """
        raise NotImplementedError()

    def decode_token(self, token: int) -> str:
        """
        Converts a single token ID to a string.
        """
        raise NotImplementedError()


class TikTokenTokenizer(BaseTokenizer):
    """
    Tokenizer for TikToken.
    """

    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")
        # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
        self.vocab_size = 50257

    def encode(self, text) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode_sequence(self, tokens: list[int]) -> str:
        # Note that multiple tokens may have been used to represent a single UTF-8 character.
        return self.encoding.decode(tokens)

    def decode_token(self, token: int) -> str:
        # Decoding falls back to a hex representation if the result isn't printable.
        s = self.encoding.decode_single_token_bytes(token)
        try:
            return s.decode("utf-8")
        except Exception as _:
            return "".join([f"\\x{d:02x}" for d in s])


class ASCIITokenizer(BaseTokenizer):
    """
    Tokenizer that treats each character in the input as a token.
    """

    def __init__(self):
        self.vocab_size = 128

    def encode(self, text) -> list[int]:
        # Note that invalid characters are replaced with "?".
        return [ord(c) if ord(c) < 128 else ord("?") for c in text]

    def decode_sequence(self, tokens: list[int]) -> str:
        return "".join([chr(b) for b in tokens])

    def decode_token(self, token: int) -> str:
        return chr(token)
