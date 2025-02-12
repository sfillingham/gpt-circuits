from typing import Sequence

import torch

from circuits.features import Feature
from circuits.search.ablation import Ablator
from circuits.search.divergence import analyze_divergence, get_predictions
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class NodeSearch:
    """
    Search for circuit nodes in a sparsified model.
    """

    def __init__(self, model: SparsifiedGPT, ablator: Ablator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param ablator: Ablation tecnique to use for circuit extraction.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.ablator = ablator
        self.num_samples = num_samples

    def search(
        self,
        tokens: Sequence[int],
        layer_idx: int,
        start_token_idx: int,
        target_token_idx: int,
        threshold: float,
    ) -> frozenset[Feature]:
        """
        Search for circuit nodes in a sparsified model.

        :param tokens: The token inputs.
        :param layer_idx: The layer index to search in.
        :param start_token_idx: The token index to start search from.
        :param target_token_idx: The target token index.
        :param threshold: The KL diverence threshold for node extraction.
        """
        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)
        tokenizer = self.model.gpt.config.tokenizer

        # Get target logits
        with torch.no_grad():
            output: SparsifiedGPTOutput = self.model(input)
        target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
        target_predictions = get_predictions(tokenizer, target_logits)
        print(f"Target predictions: {target_predictions}")

        # Get baseline KL divergence
        x_reconstructed = self.model.saes[str(layer_idx)].decode(output.feature_magnitudes[layer_idx])  # type: ignore
        predicted_logits = self.model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx)
        predicted_logits = predicted_logits[0, target_token_idx, :]  # Shape: (V)
        baseline_predictions = get_predictions(tokenizer, predicted_logits)
        baseline_kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(target_logits, dim=-1),
            torch.nn.functional.softmax(predicted_logits, dim=-1),
            reduction="sum",
        )
        print(f"Baseline predictions: {baseline_predictions}")
        print(f"Baseline KL divergence: {baseline_kl_div.item():.4f}\n")

        # Get output for layer
        feature_magnitudes = output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)

        # Get non-zero features where token index is in [start_token_idx...target_token_idx]
        initial_features: set[Feature] = set({})
        non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
        assert start_token_idx <= target_token_idx
        for t, f in zip(*non_zero_indices):
            if t >= start_token_idx and t <= target_token_idx:
                initial_features.add(Feature(layer_idx, t.item(), f.item()))

        # Circuit to start pruning
        circuit_features: set[Feature] = initial_features

        ### Part 1: Start by searching for important tokens
        print("Starting search for important tokens...")

        # Group features by token index
        features_by_token_idx: dict[int, set[Feature]] = {}
        for token_idx in range(target_token_idx + 1):
            features_by_token_idx[token_idx] = set({f for f in initial_features if f.token_idx == token_idx})

        # Starting search states
        search_threshold = threshold / 2  # Use a lower threshold for coarse search
        discard_candidates: set[Feature] = set({})
        circuit_kl_div: float = float("inf")

        # Start search
        for search_step in range(target_token_idx + 1):
            # Compute KL divergence
            circuit_candidate = frozenset(circuit_features - discard_candidates)
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                [circuit_candidate],
                num_samples=self.num_samples,
            )[circuit_candidate]
            circuit_kl_div = circuit_analysis.kl_divergence
            num_unique_tokens = len(set(f.token_idx for f in circuit_candidate))

            # Print results
            print(
                f"Tokens: {num_unique_tokens}/{target_token_idx + 1} - "
                f"KL Div: {circuit_kl_div:.4f} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # If below threshold, continue search
            if circuit_kl_div < search_threshold:
                # Update candidate features
                circuit_features = set(circuit_candidate)

                # Sort features by KL divergence (descending)
                estimated_token_ablation_effects = self.estimate_token_ablation_effects(
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    feature_magnitudes,
                    circuit_features=circuit_features,
                )
                least_important_token_idx = min(estimated_token_ablation_effects.items(), key=lambda x: x[1])[0]
                least_important_token_kl_div = estimated_token_ablation_effects[least_important_token_idx]
                discard_candidates = features_by_token_idx[least_important_token_idx]

                # Check for early stopping
                if least_important_token_kl_div > search_threshold:
                    print("Stopping search - can't improve KL divergence.")
                    break

            # If above threshold, stop search
            else:
                print("Stopping search - KL divergence is too high.")
                break

        # Print results (grouped by token_idx)
        print(f"\nCircuit after token search ({len(circuit_features)}):")
        for token_idx in range(max([f.token_idx for f in circuit_features]) + 1):
            features = [f for f in circuit_features if f.token_idx == token_idx]
            if len(features) > 0:
                print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in features])}")
        print("")

        ### Part 2: Search for important features
        print("Starting search for important features...")

        # Starting search states
        search_threshold = threshold  # Use full threshold for fine-grained search
        discard_candidates: set[Feature] = set({})
        circuit_kl_div: float = float("inf")

        # # Start search
        for _ in range(len(circuit_features)):
            # Compute KL divergence
            circuit_candidate = frozenset(circuit_features - discard_candidates)
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                [circuit_candidate],
                num_samples=self.num_samples,
            )[circuit_candidate]
            circuit_kl_div = circuit_analysis.kl_divergence

            # Print results
            print(
                f"Features: {len(circuit_candidate)}/{len(initial_features)} - "
                f"KL Div: {circuit_kl_div:.4f} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # If below threshold, continue search
            if circuit_kl_div < search_threshold:
                # Update candidate features
                circuit_features = set(circuit_candidate)

                # Sort features by KL divergence (descending)
                estimated_feature_ablation_effects = self.estimate_feature_ablation_effects(
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    feature_magnitudes,
                    circuit_features=circuit_features,
                )
                least_important_feature = min(estimated_feature_ablation_effects.items(), key=lambda x: x[1])[0]
                least_important_feature_kl_div = estimated_feature_ablation_effects[least_important_feature]
                discard_candidates = {least_important_feature}

                # Check for early stopping
                if least_important_feature_kl_div > search_threshold:
                    print("Stopping search - can't improve KL divergence.")
                    break

            # If above threshold, stop search
            else:
                print("Stopping search - KL divergence is too high.")
                break

        # Print final results (grouped by token_idx)
        print(f"\nCircuit after feature search ({len(circuit_features)}):")
        for token_idx in range(max([f.token_idx for f in circuit_features]) + 1):
            features = [f for f in circuit_features if f.token_idx == token_idx]
            if len(features) > 0:
                print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in features])}")

        # Return circuit
        return frozenset(circuit_features)

    def estimate_feature_ablation_effects(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_features: set[Feature],
    ) -> dict[Feature, float]:
        """
        Map features to KL divergence.
        """
        # Generate all variations with one feature removed
        circuit_variants: dict[Feature, frozenset[Feature]] = {}
        for feature in circuit_features:
            circuit_variants[feature] = frozenset([f for f in circuit_features if f != feature])

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            [variant for variant in circuit_variants.values()],
            self.num_samples,
        )

        # Map features to KL divergence
        return {feature: kld_results[variant].kl_divergence for feature, variant in circuit_variants.items()}

    def estimate_token_ablation_effects(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_features: set[Feature],
    ) -> dict[int, float]:
        """
        Map features to KL divergence.
        """
        # Generate all variations with one token removed
        circuit_variants: dict[int, frozenset[Feature]] = {}
        unique_token_indices = {f.token_idx for f in circuit_features}
        for token_idx in unique_token_indices:
            circuit_variant = frozenset([f for f in circuit_features if f.token_idx != token_idx])
            circuit_variants[token_idx] = circuit_variant

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            [variant for variant in circuit_variants.values()],
            self.num_samples,
        )

        # Map token indices to KL divergence
        return {token_idx: kld_results[variant].kl_divergence for token_idx, variant in circuit_variants.items()}
