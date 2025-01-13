from typing import Optional, Protocol

import torch
from torch.optim import Optimizer

from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput
from training import Trainer
from training.gpt import GPTTrainer


class SAETrainer(Trainer):
    """
    Base class for sparsified GPT trainers.
    """

    def calculate_loss(self, x, y, is_eval) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """
        Calculate model loss.
        """
        output: SparsifiedGPTOutput = self.model(x, y, is_eval=is_eval)
        loss = self.output_to_loss(output)
        metrics = None

        # Only include metrics if in evaluation mode
        if is_eval:
            l0s = torch.stack([loss_components.l0 for loss_components in output.sae_loss_components.values()])
            metrics = {
                "loss": loss,
                "ce_loss": output.cross_entropy_loss,
                "sae_loss": output.sae_loss,
                "ce_loss_increases": output.ce_loss_increases,
                "l0s": l0s,
            }

        return loss, metrics

    def output_to_loss(self, output: SparsifiedGPTOutput) -> torch.Tensor:
        """
        Convert model output to loss.
        """
        ...

    def configure_optimizer(self, model: SparsifiedGPT) -> Optimizer:
        """
        Configure the optimizer for training a sparsified GPT model.
        """
        # Get existing param groups for GPT model.
        gpt_param_groups = GPTTrainer.get_param_groups(model.gpt, self.config)

        # Add SAE parameters to the optimizer.
        sae_params = [p for p in model.saes.parameters() if p.requires_grad]
        num_gpt_params = sum(p.numel() for g in gpt_param_groups for p in g["params"])
        num_sae_params = sum(p.numel() for p in sae_params)

        # Print number of parameters
        if self.is_main_process:
            print(f"Trainable GPT parameters: {num_gpt_params:,}")
            print(f"Trainable SAE parameters: {num_sae_params:,}")

        # We set weight_decay to 0.0 for SAE parameters.
        param_groups = gpt_param_groups + [{"params": sae_params, "weight_decay": 0.0}]

        # Create optimizer
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.is_fused_adamW_available,
        )
