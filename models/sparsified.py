import dataclasses
import inspect
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model

from config.sae import LossCoefficients, SAEConfig, SAEVariant
from models.gpt import GPT
from models.sae import EncoderOutput, SAELossComponents
from models.sae.gated import GatedSAE_V2


@dataclasses.dataclass
class SparsifiedGPTOutput:
    """
    Output from the forward pass of a sparsified GPT model.
    """

    logits: torch.Tensor
    cross_entropy_loss: torch.Tensor
    sae_loss_components: dict[int, SAELossComponents]

    @property
    def sae_loss(self) -> torch.Tensor:
        """
        Mean SAE loss across all trainable SAE layers.
        """
        reconstruct_losses = torch.stack([loss.reconstruct for loss in self.sae_loss_components.values()])
        sparsity_losses = torch.stack([loss.sparsity for loss in self.sae_loss_components.values()])
        aux_losses = torch.stack([loss.aux for loss in self.sae_loss_components.values()])
        return (reconstruct_losses + sparsity_losses + aux_losses).mean()


class SparsifiedGPT(nn.Module):
    """
    GPT Model with sparsified activations using sparse autoencoders.
    """

    def __init__(
        self,
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__()
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)

        # Construct sae layers
        sae_class = self.get_sae_class(config)
        layers_to_load = trainable_layers if trainable_layers else list(range(len(config.n_features)))
        self.saes = nn.ModuleDict(dict([(f"{i}", sae_class(i, config, loss_coefficients)) for i in layers_to_load]))

        # Add pre-hooks
        self.hooks = {}
        for layer_idx in layers_to_load:
            target = self.get_pre_hook_target(layer_idx)
            self.hooks[layer_idx] = target.register_forward_pre_hook(self.create_pre_hook(layer_idx))

    def forward(self, idx, targets=None) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.
        """
        # Encoders states are stored in `encoder_outputs` using hooks.
        self.encoder_outputs: dict[int, EncoderOutput] = {}
        logits, cross_entropy_loss = self.gpt(idx, targets)

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            sae_loss_components={i: output.loss for i, output in self.encoder_outputs.items() if output.loss},
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
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device="cpu"):
        """
        Load a sparsified GPT model from a directory.
        """
        # Load SAE config
        meta_path = os.path.join(dir, "sae.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Create model using saved config
        model = SparsifiedGPT(SAEConfig(**meta["config"]), loss_coefficients, trainable_layers)

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
            case SAEVariant.GATED_V2:
                return GatedSAE_V2
            case _:
                raise ValueError(f"Unrecognized SAE variant: {self.sae_variant}")

    def configure_optimizers(self, weight_decay, learning_rate, device_type, is_master_process):
        """
        Configure optimizer for the sparsified model.
        """
        # Get existing param groups from GPT optimizer
        gpt_optimizer = self.gpt.configure_optimizers(weight_decay, learning_rate, device_type, False)
        param_groups = gpt_optimizer.param_groups

        # Add SAE parameters to the optimizer
        # NOTE: We set weight_decay to 0.0 for SAE parameters
        sae_params = list(self.saes.parameters())
        num_gpt_params = sum(p.numel() for g in gpt_optimizer.param_groups for p in g["params"])
        num_sae_params = sum(p.numel() for p in sae_params)
        if is_master_process:
            print(f"Num GPT parameters: {num_gpt_params:,}")
            print(f"Num SAE parameters: {num_sae_params:,}")
        param_groups = gpt_optimizer.param_groups + [{"params": sae_params, "weight_decay": 0.0}]

        # Create new optimizer
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if is_master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
