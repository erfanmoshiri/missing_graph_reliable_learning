import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np


def compute_ppr_matrix(edge_index, num_nodes, alpha=0.15, max_iter=10, eps=1e-8):
    """
    Compute Personalized PageRank (PPR) matrix via power iteration.

    Args:
        edge_index: [2, n_edges] graph edges
        num_nodes: number of nodes
        alpha: teleport probability (default 0.15)
        max_iter: maximum number of iterations
        eps: convergence tolerance

    Returns:
        P: [num_nodes, num_nodes] PPR matrix
    """
    device = edge_index.device

    # Convert to dense adjacency and normalize
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [n, n]

    # Compute degree matrix
    deg = A.sum(dim=1)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    D_inv = torch.diag(deg_inv)

    # Normalized adjacency: A_tilde = D^{-1} A
    A_tilde = D_inv @ A

    # Initialize P with identity
    P = torch.eye(num_nodes, device=device)

    # Power iteration: P^(t+1) = (1-alpha) * A_tilde * P^(t) + alpha * I
    for _ in range(max_iter):
        P_new = (1 - alpha) * A_tilde @ P + alpha * torch.eye(num_nodes, device=device)

        # Check convergence
        if torch.norm(P_new - P, p='fro') < eps:
            break

        P = P_new

    return P


def ppr_topk_graph(edge_index, num_nodes, k=10, alpha=0.15, max_iter=10):
    """
    Create a PPR-based global view by keeping top-k PPR neighbors for each node.

    Args:
        edge_index: [2, n_edges] base graph edges
        num_nodes: number of nodes
        k: number of top PPR neighbors to keep per node
        alpha: PPR teleport probability
        max_iter: PPR iteration count

    Returns:
        ppr_edge_index: [2, n_edges_ppr] PPR-based graph edges
    """
    device = edge_index.device

    # Compute full PPR matrix
    P = compute_ppr_matrix(edge_index, num_nodes, alpha=alpha, max_iter=max_iter)

    # For each node, keep top-k PPR neighbors (excluding self)
    A_ppr = torch.zeros_like(P)

    for i in range(num_nodes):
        # Set self-connection to -inf to exclude it
        ppr_scores = P[i].clone()
        ppr_scores[i] = -float('inf')

        # Get top-k indices
        if k < num_nodes:
            topk_values, topk_indices = torch.topk(ppr_scores, k=min(k, num_nodes - 1))
            A_ppr[i, topk_indices] = 1.0
        else:
            # Keep all if k is larger than nodes
            A_ppr[i] = (ppr_scores > -float('inf')).float()

    # Convert to edge_index format
    ppr_edge_index, _ = dense_to_sparse(A_ppr)

    return ppr_edge_index


def approximate_ppr_sparse(edge_index, num_nodes, alpha=0.15, eps=1e-4, max_iter=50):
    """
    Approximate PPR using push-based algorithm (more efficient for large graphs).

    Args:
        edge_index: [2, n_edges] graph edges
        num_nodes: number of nodes
        alpha: teleport probability
        eps: threshold for approximate computation
        max_iter: maximum iterations

    Returns:
        P_approx: [num_nodes, num_nodes] approximate PPR matrix
    """
    device = edge_index.device

    # Convert to dense for simplicity (for large graphs, use sparse operations)
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

    # Degree normalization
    deg = A.sum(dim=1)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    D_inv = torch.diag(deg_inv)
    A_norm = D_inv @ A

    # Initialize
    P = torch.zeros((num_nodes, num_nodes), device=device)
    residual = torch.eye(num_nodes, device=device)

    for iteration in range(max_iter):
        # Nodes with residual > eps
        active_mask = (residual.abs().max(dim=1)[0] > eps)

        if not active_mask.any():
            break

        # Push operation
        push_mass = alpha * residual
        P = P + push_mass

        # Propagate residual
        residual = (1 - alpha) * (A_norm @ residual)

    return P


class PPRViewConstructor(torch.nn.Module):
    """
    Module to construct and cache PPR-based global view.
    """

    def __init__(self, k=10, alpha=0.15, max_iter=10, use_approximate=False):
        """
        Args:
            k: number of top PPR neighbors per node
            alpha: PPR teleport probability
            max_iter: maximum PPR iterations
            use_approximate: whether to use approximate PPR (faster for large graphs)
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_approximate = use_approximate
        self.cached_ppr_edge_index = None

    def forward(self, edge_index, num_nodes, force_recompute=False):
        """
        Construct PPR-based global view.

        Args:
            edge_index: [2, n_edges] base graph edges
            num_nodes: number of nodes
            force_recompute: whether to recompute even if cached

        Returns:
            ppr_edge_index: [2, n_edges_ppr] PPR graph edges
        """
        if self.cached_ppr_edge_index is not None and not force_recompute:
            return self.cached_ppr_edge_index

        if self.use_approximate:
            P = approximate_ppr_sparse(edge_index, num_nodes,
                                      alpha=self.alpha, max_iter=self.max_iter)
        else:
            P = compute_ppr_matrix(edge_index, num_nodes,
                                  alpha=self.alpha, max_iter=self.max_iter)

        # Keep top-k per node
        A_ppr = torch.zeros_like(P)
        for i in range(num_nodes):
            ppr_scores = P[i].clone()
            ppr_scores[i] = -float('inf')  # Exclude self

            topk_indices = torch.topk(ppr_scores, k=min(self.k, num_nodes - 1))[1]
            A_ppr[i, topk_indices] = 1.0

        ppr_edge_index, _ = dense_to_sparse(A_ppr)
        self.cached_ppr_edge_index = ppr_edge_index

        return ppr_edge_index


if __name__ == '__main__':
    from datasets import PublicGraphDataset

    print("Testing PPR view construction...")
    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    print(f"Base graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # Test PPR computation
    print("\nComputing PPR matrix...")
    P = compute_ppr_matrix(data.edge_index, data.num_nodes, alpha=0.15, max_iter=10)
    print(f"PPR matrix shape: {P.shape}")
    print(f"PPR matrix sparsity: {(P > 1e-3).float().mean():.3f}")

    # Test top-k PPR graph
    print("\nConstructing top-k PPR graph...")
    ppr_edge_index = ppr_topk_graph(data.edge_index, data.num_nodes, k=10, alpha=0.15)
    print(f"PPR graph edges: {ppr_edge_index.shape[1]}")
    print(f"Avg PPR neighbors per node: {ppr_edge_index.shape[1] / data.num_nodes:.1f}")

    # Test PPR constructor module
    print("\nTesting PPR constructor module...")
    ppr_constructor = PPRViewConstructor(k=10, alpha=0.15, max_iter=10)
    ppr_edges = ppr_constructor(data.edge_index, data.num_nodes)
    print(f"PPR edges from module: {ppr_edges.shape[1]}")

    # Test caching
    ppr_edges_cached = ppr_constructor(data.edge_index, data.num_nodes)
    print(f"Cached result matches: {torch.equal(ppr_edges, ppr_edges_cached)}")

