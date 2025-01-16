from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch.nn import functional as F

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients


@dataclass
class SAELossComponents:
    """
    Loss components for a sparse autoencoder.
    """

    reconstruct: torch.Tensor
    sparsity: torch.Tensor
    aux: torch.Tensor
    l0: torch.Tensor
    stream_l1: torch.Tensor  # L1 of residual stream (useful for analysis)

    def __init__(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        sparsity: torch.Tensor,
        aux: torch.Tensor = torch.tensor(0.0),
    ):
        self.reconstruct = F.mse_loss(x, x_reconstructed)
        self.sparsity = sparsity
        self.aux = aux
        self.l0 = (feature_magnitudes != 0).sum(dim=-1).float().mean()
        self.stream_l1 = F.l1_loss(x, torch.zeros_like(x))

    @property
    def total(self) -> torch.Tensor:
        """
        Returns sum of reconstruction, sparsity, and aux loss.
        """
        return self.reconstruct + self.sparsity + self.aux


@dataclass
class EncoderOutput:
    """
    Output from the forward pass through an SAE module.
    """

    reconstructed_activations: torch.Tensor
    feature_magnitudes: torch.Tensor
    loss: Optional[SAELossComponents] = None


class SparseAutoencoder(Protocol):
    """
    Interface for a sparse autoencoder.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        """
        Initialize the sparse autoencoder.
        """
        ...

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Forward pass of the encoder.

        x: input tensor (batch_size, ...)
        """
        ...
