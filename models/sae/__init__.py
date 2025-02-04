from dataclasses import dataclass
from typing import Optional, Protocol

import torch

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
    x_norm: torch.Tensor  # L2 norm of input (useful for experiments)
    x_l1: torch.Tensor  # L1 of residual stream (useful for analytics)

    def __init__(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        sparsity: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ):
        self.reconstruct = (x - x_reconstructed).pow(2).sum(dim=-1).mean()
        self.sparsity = sparsity
        self.aux = aux if aux is not None else torch.tensor(0.0, device=x.device)
        self.l0 = (feature_magnitudes != 0).float().sum(dim=-1).mean()
        self.x_norm = torch.norm(x, p=2, dim=-1).mean()
        self.x_l1 = torch.norm(x, p=1, dim=-1).mean()

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

        x: input tensor (B, T, embedding size)
        """
        ...

    def decode(self, feature_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        :param feature_magnitudes: SAE activations (B, T, feature size)
        :return: reconstructed activations (B, T, embedding size)
        """
        ...
