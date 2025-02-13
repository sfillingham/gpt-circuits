from collections import defaultdict

import torch

from circuits import Circuit, Edge, Node
from circuits.search.ablation import Ablator
from circuits.search.divergence import (
    analyze_divergence,
    compute_downstream_magnitudes,
    patch_feature_magnitudes,
)
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class EdgeSearch:
    """
    Search for circuit edges in a sparsified model.
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
        tokens: list[int],
        target_token_idx: int,
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
        threshold: float,
    ) -> frozenset[Edge]:
        """
        Search for circuit edges in a sparsified model.

        :param tokens: The token sequence to use for circuit extraction.
        :param upstream_nodes: The upstream nodes to use for circuit extraction.
        :param downstream_nodes: The downstream nodes to use for circuit extraction.
        :param threshold: The threshold to use for circuit extraction.
        """
        assert len(downstream_nodes) > 0
        downstream_idx = next(iter(downstream_nodes)).layer_idx
        upstream_idx = downstream_idx - 1

        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get target logits
        with torch.no_grad():
            output: SparsifiedGPTOutput = self.model(input)
        target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)

        # Get feature magnitudes
        upstream_magnitudes = output.feature_magnitudes[upstream_idx].squeeze(0)  # Shape: (T, F)
        original_downstream_magnitudes = output.feature_magnitudes[downstream_idx].squeeze(0)  # Shape: (T, F)

        # Set initial edges as all edges that could exist between upstream and downstream nodes
        initial_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    initial_edges.add(Edge(upstream, downstream))

        # Starting search states
        circuit_edges: frozenset[Edge] = frozenset(initial_edges)  # Circuit to start pruning
        discard_candidates: frozenset[Edge] = frozenset()
        circuit_kl_div: float = float("inf")

        # Start search
        for _ in range(len(initial_edges)):
            # Derive downstream magnitudes from upstream magnitudes and edges
            circuit_candidate = Circuit(downstream_nodes, edges=frozenset(circuit_edges - discard_candidates))
            downstream_magnitudes = self.derive_downstream_magnitudes(
                downstream_nodes,
                circuit_candidate.edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )

            # Compute KL divergence
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                downstream_idx,
                target_token_idx,
                target_logits,
                [circuit_candidate],
                [downstream_magnitudes],
                num_samples=self.num_samples,
            )[circuit_candidate]
            circuit_kl_div = circuit_analysis.kl_divergence

            # Print results
            print(
                f"Edges: {len(circuit_candidate.edges)}/{len(initial_edges)} - "
                f"KL Div: {circuit_kl_div:.4f} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # If below threshold, continue search
            if circuit_kl_div < threshold:
                # Update circuit
                circuit_edges = circuit_candidate.edges

                # Sort edges by KL divergence (descending)
                estimated_edge_ablation_effects = self.estimate_edge_ablation_effects(
                    downstream_nodes,
                    circuit_edges,
                    upstream_magnitudes,
                    original_downstream_magnitudes,
                    target_token_idx,
                    target_logits,
                )
                least_important_edge = min(estimated_edge_ablation_effects.items(), key=lambda x: x[1])[0]
                least_important_edge_kl_div = estimated_edge_ablation_effects[least_important_edge]
                discard_candidates = frozenset({least_important_edge})

                # Check for early stopping
                if least_important_edge_kl_div > threshold:
                    print("Stopping search - can't improve KL divergence.")
                    break

            # If above threshold, stop search
            else:
                print("Stopping search - KL divergence is too high.")
                break

        # Print final edges (grouped by downstream token)
        print(f"\nCircuit after edge search ({len(circuit_edges)}):")
        for downstream_node in sorted(downstream_nodes):
            edges = [edge for edge in circuit_edges if edge.downstream == downstream_node]
            print(f"{downstream_node}: {', '.join([str(edge.upstream) for edge in sorted(edges)])}")

        return circuit_edges

    def derive_downstream_magnitudes(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> torch.Tensor:  # Shape: (T, F)
        """
        Derive downstream feature magnitudes from upstream feature magnitudes and edges.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        :param target_token_idx: The target token index.
        """
        # Map downstream nodes to upstream dependencies
        node_to_dependencies: dict[Node, frozenset[Node]] = {}
        for node in downstream_nodes:
            node_to_dependencies[node] = frozenset([edge.upstream for edge in edges if edge.downstream == node])
        dependencies_to_nodes: dict[frozenset[Node], set[Node]] = defaultdict(set)
        for node, dependencies in node_to_dependencies.items():
            dependencies_to_nodes[dependencies].add(node)

        # Patch upstream feature magnitudes for each set of dependencies
        circuit_variants = [Circuit(nodes=dependencies) for dependencies in dependencies_to_nodes.keys()]
        upstream_idx = next(iter(downstream_nodes)).layer_idx - 1
        patched_upstream_magnitudes = patch_feature_magnitudes(  # Shape: (num_samples, T, F)
            self.ablator,
            upstream_idx,
            target_token_idx,
            circuit_variants,
            [upstream_magnitudes] * len(circuit_variants),
            num_samples=self.num_samples,
        )

        # Compute downstream feature magnitudes for each set of dependencies
        sampled_downstream_magnitudes = compute_downstream_magnitudes(  # Shape: (num_samples, T, F)
            self.model,
            upstream_idx,
            patched_upstream_magnitudes,
        )

        # Map each downstream node to a set of sampled feature magnitudes
        node_to_sampled_magnitudes: dict[Node, torch.Tensor] = {}
        for circuit_variant, magnitudes in sampled_downstream_magnitudes.items():
            for node in dependencies_to_nodes[circuit_variant.nodes]:
                node_to_sampled_magnitudes[node] = magnitudes[:, node.token_idx, node.feature_idx]

        # Patch downstream feature magnitudes using an average of the sampled feature magnitudes
        patched_downstream_magnitudes = original_downstream_magnitudes.clone()
        for node, sampled_magnitudes in node_to_sampled_magnitudes.items():
            patched_downstream_magnitudes[node.token_idx, node.feature_idx] = sampled_magnitudes.mean(dim=0)

        return patched_downstream_magnitudes

    def estimate_edge_ablation_effects(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
        target_logits: torch.Tensor,  # Shape: (V)
    ) -> dict[Edge, float]:
        """
        Estimate the KL divergence that results from ablating each edge in a circuit.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        :param target_token_idx: The target token index.
        """

        # Create a set of circuit variants with one edge removed
        circuit_variants: list[Circuit] = []
        edge_to_circuit_variant: dict[Edge, Circuit] = {}
        for edge in edges:
            circuit_variant = Circuit(downstream_nodes, edges=frozenset(edges - {edge}))
            circuit_variants.append(circuit_variant)
            edge_to_circuit_variant[edge] = circuit_variant

        # Compute downstream feature magnitudes for each circuit variant
        downstream_magnitudes: list[torch.Tensor] = []
        downstream_idx = next(iter(downstream_nodes)).layer_idx
        for circuit_variant in circuit_variants:
            magnitudes = self.derive_downstream_magnitudes(
                downstream_nodes,
                circuit_variant.edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )
            downstream_magnitudes.append(magnitudes)

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            downstream_idx,
            target_token_idx,
            target_logits,
            circuit_variants,
            downstream_magnitudes,
            self.num_samples,
        )

        # Map edges to KL divergence
        return {edge: kld_results[variant].kl_divergence for edge, variant in edge_to_circuit_variant.items()}
