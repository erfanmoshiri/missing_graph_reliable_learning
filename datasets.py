import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures
import os


class PublicGraphDataset:
    """Loader for public graph datasets: Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo"""

    def __init__(self, name, root='./data', normalize=True):
        """
        Args:
            name: 'cora', 'citeseer', 'pubmed', 'computers', 'photo'
            root: data directory
            normalize: whether to normalize features
        """
        self.name = name.lower()
        self.root = root

        transform = NormalizeFeatures() if normalize else None

        if self.name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=os.path.join(root, 'Planetoid'),
                              name=self.name.capitalize(),
                              transform=transform)
        elif self.name == 'computers':
            dataset = Amazon(root=os.path.join(root, 'Amazon'),
                           name='Computers',
                           transform=transform)
        elif self.name == 'photo':
            dataset = Amazon(root=os.path.join(root, 'Amazon'),
                           name='Photo',
                           transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        self.dataset = dataset
        self.data = dataset[0]

    def get_data(self):
        """Returns the graph data object"""
        return self.data

    def get_split_masks(self):
        """Returns train/val/test split masks"""
        return self.data.train_mask, self.data.val_mask, self.data.test_mask

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_nodes(self):
        return self.data.num_nodes


def introduce_missingness(features, edge_index=None, mask_ratio=0.5,
                         mechanism='mcar', variant=None, seed=None):
    """
    Introduce structured missingness into node features.

    DEPRECATED: Use missingness.introduce_structured_missingness instead.
    This function is kept for backwards compatibility.

    Args:
        features: torch.Tensor of shape [n_nodes, n_features]
        edge_index: [2, n_edges] graph edges (needed for MAR degree variant)
        mask_ratio: target fraction of missing values (default 0.5)
        mechanism: 'mcar', 'mar', or 'mnar'
        variant: mechanism-specific variant (see missingness.py)
        seed: random seed for reproducibility

    Returns:
        masked_features: features with missing values set to 0
        mask: binary mask (1 = observed, 0 = missing)
    """
    from missingness import introduce_structured_missingness

    if edge_index is None:
        # Create dummy edge_index if not provided (for backwards compatibility)
        n_nodes = features.shape[0]
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    mask, masked_features = introduce_structured_missingness(
        features, edge_index, mechanism=mechanism,
        target_ratio=mask_ratio, variant=variant, seed=seed
    )

    return masked_features, mask


if __name__ == '__main__':
    # Test dataset loading
    for name in ['cora', 'citeseer', 'pubmed', 'computers', 'photo']:
        print(f"\n{name.upper()}:")
        dataset = PublicGraphDataset(name)
        data = dataset.get_data()
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {dataset.num_features}")
        print(f"  Classes: {dataset.num_classes}")

        # Test missingness
        masked_features, mask = introduce_missingness(data.x, mask_ratio=0.3, mechanism='mcar', seed=42)
        print(f"  Observed ratio: {mask.mean().item():.3f}")
