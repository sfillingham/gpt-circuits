"""
Train GPT model:
$ python -m training.gpt --config=shakespeare_128x6

DDP launch for e.g. 8 GPUs:
$ torchrun --standalone --nproc_per_node=8 -m training.gpt --config=shakespeare_128x6
"""

import argparse
from typing import Optional

import torch
from torch.optim import Optimizer

from config import TrainingConfig
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

    def configure_optimizer(self, model: GPT) -> Optimizer:
        """
        Configure the optimizer for training a GPT model.
        """
        # Get parameter groups for GPT model.
        param_groups = self.get_param_groups(model, self.config, verbose=self.is_master_process)

        # Create optimizer
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.is_fused_adamW_available,
        )

    @classmethod
    def get_param_groups(cls, model: GPT, config: TrainingConfig, verbose: bool = False) -> list[dict]:
        """
        Get parameter groups for the model.
        """
        # Start with all of the candidate parameters (that require grad).
        param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Print number of parameters in each group.
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        return optim_groups


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
