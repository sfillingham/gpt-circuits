import torch
import torch.nn as nn
from torch.nn import functional as F

from models.sae import EncoderOutput, SparseAutoencoder


class BaseGatedSAE(SparseAutoencoder):
    """
    Gated Sparse Autoencoder module.
    https://arxiv.org/abs/2404.16014
    """

    def __init__(self, config, layer_idx):
        """
        n_embd: GPT embedding size.
        F: SAE dictionary size.
        """
        super().__init__()
        F = config.n_features[layer_idx]
        self.l1_coefficient = config.l1_coefficients[layer_idx]
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(F, config.n_embd)))
        self.b_gate = nn.Parameter(torch.zeros(F))
        self.b_mag = nn.Parameter(torch.zeros(F))
        self.b_dec = nn.Parameter(torch.zeros(config.n_embd))

    def get_W_gate(self):
        """
        To be implemented by derived classes.
        """
        raise NotImplementedError()

    def get_W_mag(self):
        """
        To be implemented by derived classes.
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pi_gate = x_centered @ self.get_W_gate() + self.b_gate

        f_gate = (pi_gate > 0).float()  # whether to gate the feature
        f_mag = F.relu(x_centered @ self.get_W_mag() + self.b_mag)  # feature magnitudes

        x_encoded = f_gate * f_mag

        return x_encoded, pi_gate

    def decode(self, x: torch.Tensor):
        """
        x: SAE activations (batch_size, F)
        """
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Returns (i) a reconstruction of GPT model activations, (ii) the SAE activations, and (iii) SAE loss components.

        x: GPT model activations (batch_size, n_embd)
        """
        feature_magnitudes, pi_gate = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)

        # L2 reconstruction loss
        reconstruction_loss = F.mse_loss(x_reconstructed, x)

        # Use Gated (RI-L1) sparsity variant: https://arxiv.org/pdf/2407.14435
        scaled_pi_gate = F.relu(pi_gate) * self.W_dec.data.norm(dim=1)
        sparsity_loss = F.l1_loss(scaled_pi_gate, torch.zeros_like(pi_gate)) * self.l1_coefficient

        # compute the auxiliary loss
        W_dec_clone = self.W_dec.clone().detach()
        b_dec_clone = self.b_dec.clone().detach()
        x_hat_frozen = nn.ReLU()(pi_gate) @ W_dec_clone + b_dec_clone
        aux_loss = F.mse_loss(x_hat_frozen, x)

        # L0 sparsity loss
        l0 = (feature_magnitudes != 0).sum(dim=-1).float().mean()

        return EncoderOutput(x_reconstructed, feature_magnitudes, reconstruction_loss, sparsity_loss, aux_loss, l0)


class GatedSAE(BaseGatedSAE):
    """
    Standed Gated Sparse Autoencoder module (RI-L1).
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        F = config.n_features[layer_idx]
        self.W_gate = nn.Parameter(self.W_dec.mT.detach().clone())
        self.r_mag = nn.Parameter(torch.zeros(F))

    def get_W_gate(self):
        return self.W_gate

    def get_W_mag(self):
        return self.get_W_gate() * torch.exp(self.r_mag)


class GatedSAE_V2(BaseGatedSAE):
    """
    Experimental Gated Sparse Autoencoder module that ties the encoder and decoder weights to avoid feature absorption.
    Reference: https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN
    """

    def get_W_gate(self):
        """
        Tying encoder weights to decoder weights.
        """
        return self.W_dec.t()

    def get_W_mag(self):
        """
        The r_mag scaler doesn't seem useful after weights are tied.
        """
        return self.get_W_gate()
