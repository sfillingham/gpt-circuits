import dataclasses
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from config.sae import SAEConfig
from models.gpt import GPT


class SparsifiedGPT(nn.Module):
    """
    GPT Model with sparsified activations using sparse autoencoders.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.gpt = GPT(config.gpt_config)

        # TODO: Implement sparsified activations
        self.hooks = {}
        for name, module in self.gpt.named_modules():
            print(name)

    def forward(self, idx, targets=None):
        output = self.gpt(idx, targets)
        return output

    @classmethod
    def load(cls, dir, device="cpu"):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Create model using config
        model = SparsifiedGPT(SAEConfig(**meta["config"]))
        model.gpt = GPT.load(dir, device=device)
        return model

    def save(self, dir):
        """
        Save the sparsified GPT model to a directory.
        """
        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = {"config": dataclasses.asdict(self.config)}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Save GPT model
        self.gpt.save(dir)

    def configure_optimizers(self, weight_decay, learning_rate, device_type, is_master_process):
        """
        Configure optimizer for the sparsified model.
        """
        return self.gpt.configure_optimizers(weight_decay, learning_rate, device_type, is_master_process)


@dataclass
class EncoderOutput:
    """
    Output from the forward pass of an SAE model.
    """

    reconstructed_activations: torch.Tensor
    feature_magnitudes: torch.Tensor
    reconstruct_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    aux_loss: torch.Tensor
    l0: torch.Tensor


class SparseAutoencoder(nn.Module):
    """
    Base class for a sparse autoencoder.
    """

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass of the encoder.

        x: input tensor (batch_size, ...)
        """
        raise NotImplementedError()
