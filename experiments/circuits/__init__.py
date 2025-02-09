from typing import Sequence

import torch
from tqdm import tqdm

from circuits.features import Feature
from circuits.features.ablation import Ablator
from models.sparsified import SparsifiedGPT


@torch.no_grad()
def calculate_kl_divergence(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,  # Shape: (V)
    feature_magnitudes: torch.Tensor,  # Shape: (T, F)
    circuit_features: Sequence[Feature] = (),
) -> tuple[float, dict[str, float]]:
    """
    Calculate KL divergence between target logits and logits produced by model using reconstructed activations.
    """
    # Copy feature magnitudes to avoid modifying original
    feature_magnitudes = feature_magnitudes.clone()

    # Patch feature magnitudes
    patched_feature_magnitudes = ablator.patch(
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        feature_magnitudes=feature_magnitudes,
        circuit_features=circuit_features,
        num_samples=32,
    )

    # Reconstruct activations and compute logits
    x_reconstructed = model.saes[str(layer_idx)].decode(patched_feature_magnitudes)  # type: ignore
    predicted_logits = model.gpt.forward_with_patched_activations(
        x_reconstructed, layer_idx=layer_idx
    )  # Shape: (B, T, V)

    # We only care about logits for the target token
    predicted_logits = predicted_logits[:, target_token_idx, :]  # Shape: (B, V)

    # Convert logits to probabilities before averaging across samples
    predicted_probabilities = torch.nn.functional.softmax(predicted_logits, dim=-1)
    predicted_probabilities = predicted_probabilities.mean(dim=0)  # Shape: (V)
    predicted_logits = torch.log(predicted_probabilities)  # Shape: (V)

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


def estimate_ablation_effects(
    model: SparsifiedGPT,
    ablator: Ablator,
    layer_idx: int,
    target_token_idx: int,
    target_logits: torch.Tensor,
    feature_magnitudes: torch.Tensor,
    circuit_features: list[Feature],
) -> dict[Feature, float]:
    """
    Map features to KL divergence.
    """
    feature_to_kl_div: dict[Feature, float] = {}
    for feature in tqdm(circuit_features, desc="Estimating ablation effects"):
        patched_features = [f for f in circuit_features if f != feature]
        kl_div, _ = calculate_kl_divergence(
            model,
            ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            circuit_features=patched_features,
        )
        feature_to_kl_div[feature] = kl_div
    return feature_to_kl_div
