"""
Train GPT model:
$ python -m training.gpt --config=shakespeare_128x6

DDP launch for e.g. 8 GPUs:
$ torchrun --standalone --nproc_per_node=8 -m training.gpt --config=shakespeare_128x6
"""

import argparse
from typing import Optional

import torch

from config.gpt.training import GPTTrainingConfig, options
from models.gpt import GPT
from training import Trainer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="shakespeare_64x4", help="Training config")
    parser.add_argument("--load_from", type=str, help="Path to load model from")
    return parser.parse_args()


class GPTTrainer(Trainer):
    """
    Trainer for GPT models.
    """

    def __init__(self, config: GPTTrainingConfig, load_from: str | None = None):
        """
        Load GPT model.
        """
        if load_from:
            model = GPT.load(load_from, device=self.device)
            print(f"Loaded model from checkpoint: {load_from}")
        else:
            model = GPT(config.gpt_config)

        super().__init__(model, config)

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Calculate model loss.
        """
        _, loss = self.model(x, y)

        return loss, None


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config_name = args.config
    config = options[config_name]

    # Initialize trainer
    trainer = GPTTrainer(config, args.load_from)
    trainer.train()

    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
