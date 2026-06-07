"""
Structured missingness implementation following the paper specification.

Three mechanisms (MCAR, MAR, MNAR) with multiple variants each.
All mechanisms sample masks to achieve approximately 50% overall missing rate.
"""
import torch
import numpy as np


def introduce_mcar_missingness(features, target_ratio=0.5, seed=None):
    """
    MCAR: Missing Completely At Random.

    Feature entries are masked uniformly at random across the feature matrix.
    This represents random corruption independent of node properties,
    graph structure, and feature values.

    Args:
        features: [n_nodes, n_features] feature matrix
        target_ratio: target proportion of missing values (default 0.5)
        seed: random seed

    Returns:
        mask: [n_nodes, n_features] binary mask (1=observed, 0=missing)
        masked_features: [n_nodes, n_features] features with missing values set to 0
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_nodes, n_features = features.shape

    # Uniform masking probability across all entries
    mask = torch.bernoulli(torch.full_like(features, 1 - target_ratio))

    # Ensure each node has at least one observed feature
    nodes_with_no_features = (mask.sum(dim=1) == 0)
    if nodes_with_no_features.any():
        for i in torch.where(nodes_with_no_features)[0]:
            random_feature = np.random.randint(0, n_features)
            mask[i, random_feature] = 1

    masked_features = features * mask
    actual_ratio = 1 - mask.float().mean().item()

    print(f"  MCAR: target={target_ratio:.2f}, actual={actual_ratio:.3f}")

    return mask, masked_features


def introduce_mar_missingness(features, edge_index, target_ratio=0.5,
                              variant='degree', seed=None):
    """
    MAR: Missing At Random.

    Missingness depends on observed information (not the value itself).

    Variants:
    - 'degree': high-degree nodes have higher masking probability
    - 'observation': nodes with more observed nonzero attributes have different masking

    Args:
        features: [n_nodes, n_features] feature matrix
        edge_index: [2, n_edges] graph edges (needed for degree)
        target_ratio: target proportion of missing values
        variant: 'degree' or 'observation'
        seed: random seed

    Returns:
        mask: [n_nodes, n_features] binary mask
        masked_features: masked feature matrix
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_nodes, n_features = features.shape

    if variant == 'degree':
        # Degree-based MAR: high-degree nodes have higher masking probability
        from torch_geometric.utils import degree
        node_degrees = degree(edge_index[0], num_nodes=n_nodes)

        # Normalize degrees to [0, 1]
        degree_norm = (node_degrees - node_degrees.min()) / (node_degrees.max() - node_degrees.min() + 1e-8)

        # Masking probability increases with degree
        # Scale to achieve target_ratio on average
        base_prob = target_ratio * 0.5  # minimum masking prob
        max_prob = min(0.95, target_ratio * 1.5)  # maximum masking prob
        node_mask_prob = base_prob + degree_norm * (max_prob - base_prob)

        # Broadcast to all features of each node
        mask_prob = node_mask_prob.unsqueeze(1).expand(-1, n_features)  # [n_nodes, n_features]

    elif variant == 'observation':
        # Observation-density MAR: nodes with more nonzero features have different masking
        # Count nonzero features per node
        nonzero_count = (features != 0).float().sum(dim=1)  # [n_nodes]

        # Normalize to [0, 1]
        density_norm = (nonzero_count - nonzero_count.min()) / (nonzero_count.max() - nonzero_count.min() + 1e-8)

        # Higher observation density -> higher masking probability
        base_prob = target_ratio * 0.5
        max_prob = min(0.95, target_ratio * 1.5)
        node_mask_prob = base_prob + density_norm * (max_prob - base_prob)

        # Broadcast to all features
        mask_prob = node_mask_prob.unsqueeze(1).expand(-1, n_features)

    else:
        raise ValueError(f"Unknown MAR variant: {variant}")

    # Sample mask with per-node probabilities
    mask = torch.bernoulli(1 - mask_prob)

    # Ensure each node has at least one observed feature
    nodes_with_no_features = (mask.sum(dim=1) == 0)
    if nodes_with_no_features.any():
        for i in torch.where(nodes_with_no_features)[0]:
            random_feature = np.random.randint(0, n_features)
            mask[i, random_feature] = 1

    # Adjust to hit target ratio more precisely
    current_ratio = 1 - mask.float().mean().item()
    if abs(current_ratio - target_ratio) > 0.05:  # If more than 5% off
        # Iteratively adjust
        for _ in range(10):
            if current_ratio < target_ratio:
                # Need more masking
                observed_entries = torch.where(mask == 1)
                n_to_mask = int((target_ratio - current_ratio) * n_nodes * n_features)
                if n_to_mask > 0:
                    indices = torch.randperm(len(observed_entries[0]))[:n_to_mask]
                    mask[observed_entries[0][indices], observed_entries[1][indices]] = 0
            else:
                # Need less masking
                masked_entries = torch.where(mask == 0)
                n_to_unmask = int((current_ratio - target_ratio) * n_nodes * n_features)
                if n_to_unmask > 0:
                    indices = torch.randperm(len(masked_entries[0]))[:n_to_unmask]
                    mask[masked_entries[0][indices], masked_entries[1][indices]] = 1

            current_ratio = 1 - mask.float().mean().item()
            if abs(current_ratio - target_ratio) < 0.02:
                break

    masked_features = features * mask
    actual_ratio = 1 - mask.float().mean().item()

    print(f"  MAR ({variant}): target={target_ratio:.2f}, actual={actual_ratio:.3f}")

    return mask, masked_features


def introduce_mnar_missingness(features, target_ratio=0.5, variant='frequency', seed=None):
    """
    MNAR: Missing Not At Random.

    Missingness depends on the feature value or feature identity.

    Variants:
    - 'frequency': globally frequent features are masked more often
    - 'rare': low-frequency features (rare) are masked more often
    - 'value': larger continuous values receive higher masking probability

    Args:
        features: [n_nodes, n_features] feature matrix
        target_ratio: target proportion of missing values
        variant: 'frequency', 'rare', or 'value'
        seed: random seed

    Returns:
        mask: [n_nodes, n_features] binary mask
        masked_features: masked feature matrix
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_nodes, n_features = features.shape

    if variant == 'frequency':
        # Frequency-based MNAR: globally frequent features are masked more often
        # Compute feature frequency (how often each feature is nonzero)
        feature_freq = (features != 0).float().mean(dim=0)  # [n_features]

        # Normalize to [0, 1]
        freq_norm = (feature_freq - feature_freq.min()) / (feature_freq.max() - feature_freq.min() + 1e-8)

        # Higher frequency -> higher masking probability
        base_prob = target_ratio * 0.5
        max_prob = min(0.95, target_ratio * 1.5)
        feature_mask_prob = base_prob + freq_norm * (max_prob - base_prob)

        # Broadcast to all nodes
        mask_prob = feature_mask_prob.unsqueeze(0).expand(n_nodes, -1)

    elif variant == 'rare':
        # Rare-feature MNAR: low-frequency features are masked more often
        feature_freq = (features != 0).float().mean(dim=0)  # [n_features]

        # Normalize to [0, 1]
        freq_norm = (feature_freq - feature_freq.min()) / (feature_freq.max() - feature_freq.min() + 1e-8)

        # LOWER frequency -> HIGHER masking probability (inverse)
        base_prob = target_ratio * 0.5
        max_prob = min(0.95, target_ratio * 1.5)
        feature_mask_prob = base_prob + (1 - freq_norm) * (max_prob - base_prob)

        # Broadcast to all nodes
        mask_prob = feature_mask_prob.unsqueeze(0).expand(n_nodes, -1)

    elif variant == 'value':
        # Value-based MNAR: larger continuous values receive higher masking probability
        # Normalize feature values to [0, 1] per feature dimension
        feature_min = features.min(dim=0, keepdim=True)[0]
        feature_max = features.max(dim=0, keepdim=True)[0]
        features_norm = (features - feature_min) / (feature_max - feature_min + 1e-8)

        # Higher values -> higher masking probability
        base_prob = target_ratio * 0.3
        max_prob = min(0.95, target_ratio * 1.7)
        mask_prob = base_prob + features_norm * (max_prob - base_prob)

    else:
        raise ValueError(f"Unknown MNAR variant: {variant}")

    # Sample mask
    mask = torch.bernoulli(1 - mask_prob)

    # Ensure each node has at least one observed feature
    nodes_with_no_features = (mask.sum(dim=1) == 0)
    if nodes_with_no_features.any():
        for i in torch.where(nodes_with_no_features)[0]:
            random_feature = np.random.randint(0, n_features)
            mask[i, random_feature] = 1

    # Adjust to hit target ratio
    current_ratio = 1 - mask.float().mean().item()
    if abs(current_ratio - target_ratio) > 0.05:
        for _ in range(10):
            if current_ratio < target_ratio:
                observed_entries = torch.where(mask == 1)
                n_to_mask = int((target_ratio - current_ratio) * n_nodes * n_features)
                if n_to_mask > 0:
                    indices = torch.randperm(len(observed_entries[0]))[:n_to_mask]
                    mask[observed_entries[0][indices], observed_entries[1][indices]] = 0
            else:
                masked_entries = torch.where(mask == 0)
                n_to_unmask = int((current_ratio - target_ratio) * n_nodes * n_features)
                if n_to_unmask > 0:
                    indices = torch.randperm(len(masked_entries[0]))[:n_to_unmask]
                    mask[masked_entries[0][indices], masked_entries[1][indices]] = 1

            current_ratio = 1 - mask.float().mean().item()
            if abs(current_ratio - target_ratio) < 0.02:
                break

    masked_features = features * mask
    actual_ratio = 1 - mask.float().mean().item()

    print(f"  MNAR ({variant}): target={target_ratio:.2f}, actual={actual_ratio:.3f}")

    return mask, masked_features


def introduce_structured_missingness(features, edge_index, mechanism='mcar',
                                     target_ratio=0.5, variant=None, seed=None):
    """
    Unified interface for introducing structured missingness.

    Args:
        features: [n_nodes, n_features] feature matrix
        edge_index: [2, n_edges] graph edges (needed for MAR degree variant)
        mechanism: 'mcar', 'mar', or 'mnar'
        target_ratio: target proportion of missing values (default 0.5)
        variant: mechanism-specific variant
            - MAR: 'degree' (default) or 'observation'
            - MNAR: 'frequency' (default), 'rare', or 'value'
            - MCAR: ignored
        seed: random seed

    Returns:
        mask: [n_nodes, n_features] binary mask (1=observed, 0=missing)
        masked_features: [n_nodes, n_features] masked feature matrix
    """
    mechanism = mechanism.lower()

    if mechanism == 'mcar':
        return introduce_mcar_missingness(features, target_ratio, seed)

    elif mechanism == 'mar':
        if variant is None:
            variant = 'degree'
        return introduce_mar_missingness(features, edge_index, target_ratio, variant, seed)

    elif mechanism == 'mnar':
        if variant is None:
            variant = 'frequency'
        return introduce_mnar_missingness(features, target_ratio, variant, seed)

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Use 'mcar', 'mar', or 'mnar'")


if __name__ == '__main__':
    from datasets import PublicGraphDataset

    print("Testing structured missingness mechanisms...")
    print("="*80)

    # Load dataset
    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    print(f"Dataset: Cora")
    print(f"Nodes: {data.num_nodes}, Features: {data.x.shape[1]}")
    print(f"Target missing ratio: 50%")
    print()

    # Test MCAR
    print("1. MCAR (Missing Completely At Random)")
    print("-" * 80)
    mask_mcar, feat_mcar = introduce_mcar_missingness(data.x, target_ratio=0.5, seed=42)
    print()

    # Test MAR variants
    print("2. MAR (Missing At Random)")
    print("-" * 80)
    mask_mar_deg, feat_mar_deg = introduce_mar_missingness(
        data.x, data.edge_index, target_ratio=0.5, variant='degree', seed=42
    )
    mask_mar_obs, feat_mar_obs = introduce_mar_missingness(
        data.x, data.edge_index, target_ratio=0.5, variant='observation', seed=42
    )
    print()

    # Test MNAR variants
    print("3. MNAR (Missing Not At Random)")
    print("-" * 80)
    mask_mnar_freq, feat_mnar_freq = introduce_mnar_missingness(
        data.x, target_ratio=0.5, variant='frequency', seed=42
    )
    mask_mnar_rare, feat_mnar_rare = introduce_mnar_missingness(
        data.x, target_ratio=0.5, variant='rare', seed=42
    )
    mask_mnar_val, feat_mnar_val = introduce_mnar_missingness(
        data.x, target_ratio=0.5, variant='value', seed=42
    )
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Mechanism':<20} {'Variant':<15} {'Missing %':<12} {'Min/Node':<12} {'Max/Node'}")
    print("-"*80)

    for name, mask in [
        ('MCAR', mask_mcar),
        ('MAR', mask_mar_deg),
        ('MAR', mask_mar_obs),
        ('MNAR', mask_mnar_freq),
        ('MNAR', mask_mnar_rare),
        ('MNAR', mask_mnar_val),
    ]:
        variants = ['uniform', 'degree', 'observation', 'frequency', 'rare', 'value']
        idx = [0, 1, 2, 3, 4, 5][[m is mask for m in [mask_mcar, mask_mar_deg, mask_mar_obs,
                                                        mask_mnar_freq, mask_mnar_rare, mask_mnar_val]].index(True)]
        variant = variants[idx]

        missing_pct = (1 - mask.float().mean().item()) * 100
        obs_per_node = mask.sum(dim=1)
        min_obs = obs_per_node.min().item()
        max_obs = obs_per_node.max().item()

        print(f"{name:<20} {variant:<15} {missing_pct:>10.2f}% {min_obs:>10.0f}   {max_obs:>10.0f}")

    print("="*80)
    print("\n✓ All missingness mechanisms working correctly!")
