from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.sae import LossCoefficients, SAEConfig
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class GatedSAE(nn.Module, SparseAutoencoder):
    """
    Gated sparse autoencoder with RI-L1 sparsity penalty
    https://arxiv.org/abs/2404.16014
    https://arxiv.org/html/2407.14435v3
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__()
        F = config.n_features[layer_idx]  # SAE dictionary size.
        n_embd = config.gpt_config.n_embd  # GPT embedding size.
        self.l1_coefficient = loss_coefficients.l1[layer_idx] if loss_coefficients else None
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(F, n_embd)))
        self.b_gate = nn.Parameter(torch.zeros(F))
        self.b_mag = nn.Parameter(torch.zeros(F))
        self.b_dec = nn.Parameter(torch.zeros(n_embd))

        try:
            # NOTE: Subclass might define these using properties.
            self.W_gate = nn.Parameter(self.W_dec.mT.detach().clone())
            self.r_mag = nn.Parameter(torch.zeros(F))
        except KeyError:
            pass

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: GPT model activations (batch_size, n_embd)
        """
        x_centered = x - self.b_dec
        pi_gate = x_centered @ self.W_gate + self.b_gate

        f_gate = (pi_gate > 0).float()  # whether to gate the feature
        W_mag = self.W_gate * torch.exp(self.r_mag)
        f_mag = F.relu(x_centered @ W_mag + self.b_mag)  # feature magnitudes

        feature_magnitudes = f_gate * f_mag
        return feature_magnitudes, pi_gate

    def decode(self, feature_magnitudes: torch.Tensor):
        """
        feature_magnitudes: SAE activations (batch_size, F)
        """
        return feature_magnitudes @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (batch_size, n_embd)
        """
        feature_magnitudes, pi_gate = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)

        if self.l1_coefficient:
            # Use Gated (RI-L1) sparsity variant: https://arxiv.org/pdf/2407.14435
            scaled_pi_gate = F.relu(pi_gate) * self.W_dec.data.norm(dim=1)
            sparsity_loss = F.l1_loss(scaled_pi_gate, torch.zeros_like(pi_gate)) * self.l1_coefficient

            # compute the auxiliary loss
            W_dec_clone = self.W_dec.clone().detach()
            b_dec_clone = self.b_dec.clone().detach()
            x_hat_frozen = nn.ReLU()(pi_gate) @ W_dec_clone + b_dec_clone
            aux_loss = F.mse_loss(x_hat_frozen, x)

            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss, aux_loss)

        return output


class GatedSAE_V2(GatedSAE):
    """
    Experimental Gated Sparse Autoencoder module that ties the encoder and decoder weights to avoid feature absorption.
    Reference: https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN
    """

    @property
    def W_gate(self):
        """
        Tying encoder weights to decoder weights.
        """
        return self.W_dec.t()

    @property
    def r_mag(self):
        """
        The r_mag scaler doesn't seem useful after weights are tied.
        """
        return torch.zeros_like(self.b_mag)
