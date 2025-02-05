from dataclasses import dataclass
from typing import Sequence

import torch

from models.sparsified import SparsifiedGPT


@dataclass
class MaskedFeature:
    """
    Represents a masked feature at a specific location.
    """

    token_idx: int
    feature_idx: int

    def as_tuple(self) -> tuple[int, int]:
        return self.token_idx, self.feature_idx


@torch.no_grad()
def calculate_kl_divergence(
    model: SparsifiedGPT,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    masked_features: Sequence[MaskedFeature] = (),
) -> tuple[float, dict[str, float]]:
    """
    Calculate KL divergence between target logits and logits produced by model using reconstructed activations.
    """
    # Copy feature magnitudes to avoid modifying original
    feature_magnitudes = feature_magnitudes.clone()

    # Ablate masked features
    for masked_feature in masked_features:
        feature_magnitudes[masked_feature.token_idx, masked_feature.feature_idx] = 0

    # Reconstruct activations and compute logits
    feature_magnitudes = feature_magnitudes.unsqueeze(0)  # Shape: (1, T, F)
    x_reconstructed = model.saes[str(layer_idx)].decode(feature_magnitudes)  # type: ignore
    predicted_logits = model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx)
    predicted_logits = predicted_logits.squeeze(0)[target_token_idx]  # Shape: (V)

    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(predicted_logits, dim=-1),
        torch.nn.functional.softmax(target_logits, dim=-1),
        reduction="sum",
    )

    # Calculate predictions
    predictions = get_predictions(model, predicted_logits)

    return kl_div.item(), predictions


def get_predictions(
    model: SparsifiedGPT,
    logits: torch.Tensor,  # Shape: (V)
    count: int = 5,
) -> dict[str, float]:
    """
    Map logits to probabilities and return top 5 predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=count)
    results: dict[str, float] = {}
    for i, p in zip(topk.indices, topk.values):
        results[model.gpt.config.tokenizer.decode_token(int(i.item()))] = round(p.item() * 100, 2)
    return results
