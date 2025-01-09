import dataclasses
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model

from config.sae import LossCoefficients, SAEConfig, SAEVariant
from models.gpt import GPT
from models.sae.base import EncoderOutput
from models.sae.gated import GatedSAE_V2


@dataclasses.dataclass
class SparsifiedGPTOutput:
    """
    Output from the forward pass of a sparsified GPT model.
    """

    logits: torch.Tensor
    ce_loss: torch.Tensor
    sae_reconstruct_losses: torch.Tensor
    sae_sparsity_losses: torch.Tensor
    sae_aux_losses: torch.Tensor
    sae_l0_losses: torch.Tensor

    @property
    def sae_loss(self) -> torch.Tensor:
        return (self.sae_reconstruct_losses + self.sae_sparsity_losses + self.sae_aux_losses).mean()

    @property
    def loss(self) -> torch.Tensor:
        return self.ce_loss + self.sae_loss


class SparsifiedGPT(nn.Module):
    """
    GPT Model with sparsified activations using sparse autoencoders.
    """

    def __init__(self, config: SAEConfig, loss_coefficients: Optional[LossCoefficients] = None):
        super().__init__()
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)

        # Construct sae layers
        sae_class = self.get_sae_class(config)
        target_layers = list(range(len(config.n_features)))
        self.saes = nn.ModuleDict(dict([(f"{i}", sae_class(i, config, loss_coefficients)) for i in target_layers]))

        # Add pre-hooks
        self.hooks = {}
        for layer_idx in target_layers:
            target = self.get_pre_hook_target(layer_idx)
            self.hooks[layer_idx] = target.register_forward_pre_hook(self.create_pre_hook(layer_idx))

    def forward(self, idx, targets=None) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.
        """
        # Encoders states are stored in `encoder_outputs` using hooks.
        self.encoder_outputs: dict[int, EncoderOutput] = {}
        logits, cross_entropy_loss = self.gpt(idx, targets)

        sae_reconstruct_losses = torch.stack([output.reconstruct_loss for output in self.encoder_outputs.values()])
        sae_sparsity_losses = torch.stack([output.sparsity_loss for output in self.encoder_outputs.values()])
        sae_aux_losses = torch.stack([output.aux_loss for output in self.encoder_outputs.values()])
        sae_l0_losses = torch.stack([output.l0 for output in self.encoder_outputs.values()])

        return SparsifiedGPTOutput(
            logits=logits,
            ce_loss=cross_entropy_loss,
            sae_reconstruct_losses=sae_reconstruct_losses,
            sae_sparsity_losses=sae_sparsity_losses,
            sae_aux_losses=sae_aux_losses,
            sae_l0_losses=sae_l0_losses,
        )

    def get_pre_hook_target(self, layer_idx) -> nn.Module:
        """
        SAE layer -> Targeted named module for forward pre-hook.
        """
        if layer_idx < self.config.gpt_config.n_layer:
            return self.gpt.get_submodule(f"transformer.h.{layer_idx}")
        elif layer_idx == self.config.gpt_config.n_layer:
            return self.gpt.get_submodule("transformer.ln_f")
        raise ValueError(f"Invalid layer index: {layer_idx}")

    def create_pre_hook(self, layer_idx):
        """
        Create a forward pre-hook for the given layer index.
        """

        def hook(module, inputs):
            x = inputs[0]
            sae = self.saes[f"{layer_idx}"]
            self.encoder_outputs[layer_idx] = sae(x)

        return hook

    @classmethod
    def load(cls, dir, loss_coefficients=None, device="cpu"):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Create model using saved config
        model = SparsifiedGPT(SAEConfig(**meta["config"]), loss_coefficients)

        # Load GPT weights
        model.gpt = GPT.load(dir, device=device)

        # Load SAE weights
        for layer_name, module in model.saes.items():
            weights_path = os.path.join(dir, f"sae_{layer_name}.safetensors")
            load_model(module, weights_path, device=device)

        return model

    def save(self, dir):
        """
        Save the sparsified GPT model to a directory.
        """
        # Save GPT model
        self.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = {"config": dataclasses.asdict(self.config)}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Save SAE modules
        for layer_name, module in self.saes.items():
            weights_path = os.path.join(dir, f"sae_{layer_name}.safetensors")
            save_model(module, weights_path)

    def get_sae_class(self, config: SAEConfig) -> type:
        """
        Maps the SAE variant to the actual class.
        """
        match config.sae_variant:
            case SAEVariant.GATED_V2.value:
                return GatedSAE_V2
            case _:
                raise ValueError(f"Unrecognized SAE variant: {self.sae_variant}")

    def configure_optimizers(self, weight_decay, learning_rate, device_type, is_master_process):
        """
        Configure optimizer for the sparsified model.
        """
        return self.gpt.configure_optimizers(weight_decay, learning_rate, device_type, is_master_process)
