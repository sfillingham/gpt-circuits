from circuits import Edge, Node
from circuits.search.ablation import Ablator
from models.sparsified import SparsifiedGPT


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
        upstream_nodes: set[Node],
        downstream_nodes: set[Node],
        threshold: float,
    ) -> frozenset[Edge]:
        """
        Search for circuit edges in a sparsified model.

        :param tokens: The token sequence to use for circuit extraction.
        :param upstream_nodes: The upstream nodes to use for circuit extraction.
        :param downstream_nodes: The downstream nodes to use for circuit extraction.
        :param threshold: The threshold to use for circuit extraction.
        """

        # Set initial edges as all edges that could exist between upstream and downstream nodes
        initial_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    initial_edges.add(Edge(upstream, downstream))

        print(sorted(initial_edges))
        return frozenset()
