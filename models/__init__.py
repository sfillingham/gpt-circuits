from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """
    GPT model output from forward pass.
    """

    logits: torch.Tensor
    loss: torch.Tensor = torch.Tensor()
