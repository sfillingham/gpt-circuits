from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from config.sae import LossCoefficients, SAEConfig


@dataclass
class SAELossComponents:
    """
    Loss components for a sparse autoencoder.
    """

    reconstruct: torch.Tensor
    sparsity: torch.Tensor
    aux: torch.Tensor
    l0: torch.Tensor

    def __init__(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        sparsity: torch.Tensor,
        aux: torch.Tensor,
    ):
        self.reconstruct = F.mse_loss(x, x_reconstructed)
        self.sparsity = sparsity
        self.aux = aux
        self.l0 = (feature_magnitudes != 0).sum(dim=-1).float().mean()


@dataclass
class EncoderOutput:
    """
    Output from the forward pass through an SAE module.
    """

    reconstructed_activations: torch.Tensor
    feature_magnitudes: torch.Tensor
    loss: Optional[SAELossComponents] = None


class SparseAutoencoder(ABC):
    """
    Base class for a sparse autoencoder.
    """

    @abstractmethod
    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        """
        Initialize the sparse autoencoder.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass of the encoder.

        x: input tensor (batch_size, ...)
        """
        pass
