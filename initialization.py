import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj


def mask_aware_knn_imputation(features, mask, edge_index, k=5, overlap_penalty=0.1, eps=1e-8):
    """
    Mask-aware KNN imputation using base graph neighbors.

    Args:
        features: [n_nodes, n_features] incomplete feature matrix
        mask: [n_nodes, n_features] binary mask (1=observed, 0=missing)
        edge_index: [2, n_edges] graph edges
        k: number of nearest neighbors
        overlap_penalty: lambda_o in the paper
        eps: numerical stability constant

    Returns:
        imputed_features: [n_nodes, n_features] with missing values filled
    """
    n_nodes, n_features = features.shape
    device = features.device

    # Build adjacency list
    adj = to_dense_adj(edge_index)[0]  # [n_nodes, n_nodes]

    imputed_features = features.clone()

    for i in range(n_nodes):
        # Find neighbors
        neighbors = torch.where(adj[i] > 0)[0]

        if len(neighbors) == 0 or mask[i].sum() == n_features:
            continue

        # Compute mask-aware distances
        distances = []
        valid_neighbors = []

        for j in neighbors:
            # Overlap mask
            overlap = mask[i] * mask[j]
            c_ij = overlap.sum()

            if c_ij < 1:
                continue

            # Mask-aware distance
            diff = (features[i] - features[j]) * overlap
            d_ij = (diff ** 2).sum() / (c_ij + eps)

            # Add overlap penalty
            d_ij_prime = d_ij + overlap_penalty / (c_ij + eps)

            distances.append(d_ij_prime)
            valid_neighbors.append(j)

        if len(valid_neighbors) == 0:
            continue

        # Select top-k nearest neighbors
        distances = torch.tensor(distances, device=device)
        k_actual = min(k, len(valid_neighbors))
        topk_idx = torch.topk(distances, k_actual, largest=False).indices
        selected_neighbors = [valid_neighbors[idx.item()] for idx in topk_idx]

        # Compute similarity weights (inverse distance)
        selected_distances = distances[topk_idx]
        alpha = 1.0 / (selected_distances + eps)
        alpha = alpha / (alpha.sum() + eps)

        # Impute missing features
        for f in range(n_features):
            if mask[i, f] == 0:  # Missing feature
                numerator = 0.0
                denominator = 0.0
                for idx, j in enumerate(selected_neighbors):
                    if mask[j, f] == 1:  # Neighbor has this feature
                        numerator += alpha[idx] * features[j, f]
                        denominator += alpha[idx]

                if denominator > eps:
                    imputed_features[i, f] = numerator / denominator

    return imputed_features


def compute_feature_frequencies(mask):
    """
    Compute global observation frequency for each feature.

    Args:
        mask: [n_nodes, n_features] binary mask

    Returns:
        q: [n_features] frequency vector
    """
    return mask.float().mean(dim=0)


def encode_missingness_pattern(features, mask, hidden_dim):
    """
    Encode missingness pattern as described in the paper.

    Args:
        features: [n_nodes, n_features] (imputed) feature matrix
        mask: [n_nodes, n_features] binary mask
        hidden_dim: dimension of encoded representation

    Returns:
        h0: [n_nodes, hidden_dim] missingness-aware initial embeddings
    """
    n_nodes, n_features = features.shape

    # Global feature frequencies
    q = compute_feature_frequencies(mask)  # [n_features]

    # Feature-availability pattern
    s = mask * q.unsqueeze(0)  # [n_nodes, n_features]

    # Concatenate features, mask, and availability pattern
    # [n_nodes, 3 * n_features]
    concatenated = torch.cat([features, mask.float(), s], dim=1)

    # Project to hidden dimension
    mlp = torch.nn.Sequential(
        torch.nn.Linear(3 * n_features, 2 * hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * hidden_dim, hidden_dim)
    ).to(features.device)

    h0 = mlp(concatenated)

    return h0, mlp


def compute_node_reliability(mask, edge_index, k=5, eps=1e-8):
    """
    Compute node reliability scores.

    Args:
        mask: [n_nodes, n_features] binary mask
        edge_index: [2, n_edges] graph edges
        k: number of neighbors for reliability estimation
        eps: numerical stability

    Returns:
        reliability: [n_nodes] node reliability scores
        rho: learnable parameter (initialized to 0.5)
    """
    n_nodes, n_features = mask.shape
    device = mask.device

    # Self-observation reliability
    r_self = mask.float().mean(dim=1)  # [n_nodes]

    # Build adjacency
    adj = to_dense_adj(edge_index)[0]  # [n_nodes, n_nodes]

    # Neighborhood-supported reliability
    r_nbr = torch.zeros(n_nodes, device=device)

    for i in range(n_nodes):
        neighbors = torch.where(adj[i] > 0)[0]

        if len(neighbors) == 0:
            r_nbr[i] = r_self[i]
            continue

        # Simple uniform weights for neighbors
        neighbor_obs = mask[neighbors].float().mean(dim=1).mean()
        r_nbr[i] = neighbor_obs

    # Learnable combination parameter
    rho = torch.nn.Parameter(torch.tensor(0.0, device=device))

    # Combined reliability (will be computed dynamically with learned rho)
    return r_self, r_nbr, rho


def get_reliability_score(r_self, r_nbr, rho):
    """
    Compute final reliability score with learned parameter.

    Args:
        r_self: [n_nodes] self-observation reliability
        r_nbr: [n_nodes] neighborhood reliability
        rho: learnable scalar parameter

    Returns:
        r: [n_nodes] final reliability scores
    """
    weight = torch.sigmoid(rho)
    r = weight * r_self + (1 - weight) * r_nbr
    return r


if __name__ == '__main__':
    # Test initialization components
    from datasets import PublicGraphDataset, introduce_missingness

    print("Testing initialization components...")
    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    # Introduce missingness
    masked_features, mask = introduce_missingness(data.x, mask_ratio=0.3, mechanism='mcar', seed=42)

    print(f"Original features: {data.x.shape}")
    print(f"Masked features: {masked_features.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Observed ratio: {mask.mean():.3f}")

    # Test KNN imputation
    print("\nTesting mask-aware KNN imputation...")
    imputed = mask_aware_knn_imputation(masked_features, mask, data.edge_index, k=5)
    print(f"Imputed features shape: {imputed.shape}")
    print(f"Non-zero after imputation: {(imputed != 0).float().mean():.3f}")

    # Test missingness pattern encoding
    print("\nTesting missingness pattern encoding...")
    h0, mlp = encode_missingness_pattern(imputed, mask, hidden_dim=128)
    print(f"Encoded shape: {h0.shape}")

    # Test reliability computation
    print("\nTesting node reliability...")
    r_self, r_nbr, rho = compute_node_reliability(mask, data.edge_index, k=5)
    print(f"Self reliability: mean={r_self.mean():.3f}, std={r_self.std():.3f}")
    print(f"Neighbor reliability: mean={r_nbr.mean():.3f}, std={r_nbr.std():.3f}")

    r = get_reliability_score(r_self, r_nbr, rho)
    print(f"Combined reliability: mean={r.mean():.3f}, std={r.std():.3f}")
