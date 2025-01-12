import dataclasses
import json
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model
from torch.nn import functional as F

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
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
    ce_loss_increases: Optional[torch.Tensor]
    sae_loss_components: dict[int, SAELossComponents]

    @property
    def sae_loss(self) -> torch.Tensor:
        """
        Mean SAE loss across all trainable SAE layers.
        """
        return torch.stack([loss.total for loss in self.sae_loss_components.values()]).mean()


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
        self.layer_idxs = trainable_layers if trainable_layers else list(range(len(config.n_features)))
        self.saes = nn.ModuleDict(dict([(f"{i}", sae_class(i, config, loss_coefficients)) for i in self.layer_idxs]))

    def forward(self, idx, targets=None, is_eval: bool = False) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.
        """
        with self.use_saes() as encoder_outputs:
            logits, cross_entropy_loss = self.gpt(idx, targets)

        # Calculate cross-entropy loss increase for each SAE layer if targets are provided during training evaluation.
        ce_loss_increases = None
        if targets is not None and is_eval:
            ce_loss_increases = []
            for layer_idx, output in encoder_outputs.items():
                x = output.reconstructed_activations
                sae_logits = self.gpt.forward_with_patched_activations(idx, x, layer_idx)
                sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            ce_loss_increases=ce_loss_increases,
            sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
        )

    @contextmanager
    def use_saes(self):
        """
        Context manager for using SAE layers during the forward pass.

        :yield encoder_outputs: Dictionary of encoder outputs.
        """
        # Dictionary for storing results
        encoder_outputs: dict[int, EncoderOutput] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            target = self.get_hook_target(layer_idx)
            sae = self.saes[f"{layer_idx}"]
            # Output values will be overwritten (hack to pass object by reference)
            output = EncoderOutput(torch.tensor(0), torch.tensor(0))
            hooks.append(target.register_forward_pre_hook(self.create_hook(sae, output)))
            encoder_outputs[layer_idx] = output

        try:
            yield encoder_outputs

        finally:
            # Unregister hooks
            for hook in hooks:
                hook.remove()

    def create_hook(self, sae, output):
        """
        Create a forward pre-hook for the given layer index.

        :param sae: SAE module to use for the forward pass.
        :param output: Encoder output to be updated.
        """

        def hook(_, inputs):
            x = inputs[0]
            # Override field values instead of replacing reference
            output.__dict__ = sae(x).__dict__

        return hook

    def get_hook_target(self, layer_idx) -> nn.Module:
        """
        SAE layer -> Targeted module for forward pre-hook.
        """
        if layer_idx < self.config.gpt_config.n_layer:
            return self.gpt.transformer.h[layer_idx]
        elif layer_idx == self.config.gpt_config.n_layer:
            return self.gpt.transformer.ln_f
        raise ValueError(f"Invalid layer index: {layer_idx}")

    @classmethod
    def load(cls, dir, loss_coefficients=None, trainable_layers=None, device: torch.device = torch.device("cpu")):
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
            load_model(module, weights_path, device=device.type)

        return model

    def load_gpt_weights(self, dir):
        """
        Load just the GPT model weights without loading SAE weights.
        """
        device = next(self.gpt.lm_head.parameters()).device
        self.gpt = GPT.load(dir, device=device)

    def save(self, dir, layers_to_save: Optional[list[str]] = None):
        """
        Save the sparsified GPT model to a directory.

        :param dir: Directory for saving weights.
        :param layers_to_save: Module names for SAE layers to save. If None, all layers will be saved.
        """
        # Save GPT model
        self.gpt.save(dir)

        # Save SAE config
        meta_path = os.path.join(dir, "sae.json")
        meta = {"config": dataclasses.asdict(self.config)}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Which layers should we save?
        layers_to_save = layers_to_save or list(self.saes.keys())

        # Save SAE modules
        for layer_name, module in self.saes.items():
            if layer_name in layers_to_save:
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
