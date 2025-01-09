from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from config.sae import LossCoefficients, SAEConfig


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


class SparseAutoencoder(ABC, nn.Module):
    """
    Base class for a sparse autoencoder.
    """

    @abstractmethod
    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        """
        Initialize the sparse autoencoder.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass of the encoder.

        x: input tensor (batch_size, ...)
        """
        raise NotImplementedError()
