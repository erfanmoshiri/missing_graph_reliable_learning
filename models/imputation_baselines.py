"""
Imputation baselines + GAT for node classification.

Implements simple imputation methods followed by 2-layer GAT:
- MEAN: mean imputation per feature
- KNN: k-nearest neighbors imputation
- MICE: Multiple Imputation by Chained Equations
- MissForest: iterative random forest imputation

Each method imputes missing values, then uses a 2-layer GAT for representation learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class ImputationBaseline(nn.Module):
    """
    Base class for imputation + GAT baselines.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_classes,
                 imputation_method='mean', heads=8, dropout=0.6):
        """
        Args:
            in_dim: input feature dimension
            hidden_dim: hidden dimension for GAT
            out_dim: embedding dimension
            num_classes: number of output classes
            imputation_method: 'mean', 'knn', 'mice', 'missforest'
            heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()

        self.imputation_method = imputation_method.lower()
        self.in_dim = in_dim
        self.imputer = None

        # 2-layer GAT
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)

        # Classification head
        self.classifier = nn.Linear(out_dim, num_classes)

        self.dropout = dropout

    def setup_imputer(self):
        """Initialize the imputer based on the method."""
        if self.imputation_method == 'mean':
            # Mean imputation
            self.imputer = SimpleImputer(strategy='mean')

        elif self.imputation_method == 'knn':
            # KNN imputation with k=5
            self.imputer = KNNImputer(n_neighbors=5, weights='distance')

        elif self.imputation_method == 'mice':
            # MICE (Multiple Imputation by Chained Equations)
            self.imputer = IterativeImputer(
                random_state=42,
                max_iter=10,
                tol=1e-3
            )

        elif self.imputation_method == 'missforest':
            # MissForest (iterative random forest imputation)
            self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=10,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1
                ),
                random_state=42,
                max_iter=10,
                tol=1e-3
            )

        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")

    def impute(self, features, mask):
        """
        Impute missing values in features.

        Args:
            features: [n_nodes, n_features] feature matrix (missing values as 0)
            mask: [n_nodes, n_features] binary mask (1=observed, 0=missing)

        Returns:
            imputed_features: [n_nodes, n_features] complete feature matrix
        """
        if self.imputer is None:
            self.setup_imputer()

        # Convert to numpy
        features_np = features.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Create masked array where missing values are marked as NaN
        features_with_nan = features_np.copy()
        features_with_nan[mask_np == 0] = np.nan

        # Impute
        imputed_np = self.imputer.fit_transform(features_with_nan)

        # Convert back to torch
        imputed_features = torch.from_numpy(imputed_np).float().to(features.device)

        return imputed_features

    def forward(self, features, mask, edge_index):
        """
        Forward pass: impute + GAT + classify.

        Args:
            features: [n_nodes, n_features] feature matrix
            mask: [n_nodes, n_features] observation mask
            edge_index: [2, n_edges] graph edges

        Returns:
            embeddings: [n_nodes, out_dim] node embeddings
            logits: [n_nodes, num_classes] classification logits
        """
        # Impute missing values
        with torch.no_grad():
            imputed_features = self.impute(features, mask)

        # 2-layer GAT
        x = self.gat1(imputed_features, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = self.gat2(x, edge_index)

        # Classify
        logits = self.classifier(embeddings)

        return embeddings, logits


class MeanImputationGAT(ImputationBaseline):
    """Mean imputation + GAT."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, heads=8, dropout=0.6):
        super().__init__(in_dim, hidden_dim, out_dim, num_classes,
                        imputation_method='mean', heads=heads, dropout=dropout)


class KNNImputationGAT(ImputationBaseline):
    """KNN imputation + GAT."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, heads=8, dropout=0.6):
        super().__init__(in_dim, hidden_dim, out_dim, num_classes,
                        imputation_method='knn', heads=heads, dropout=dropout)


class MICEImputationGAT(ImputationBaseline):
    """MICE imputation + GAT."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, heads=8, dropout=0.6):
        super().__init__(in_dim, hidden_dim, out_dim, num_classes,
                        imputation_method='mice', heads=heads, dropout=dropout)


class MissForestImputationGAT(ImputationBaseline):
    """MissForest imputation + GAT."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, heads=8, dropout=0.6):
        super().__init__(in_dim, hidden_dim, out_dim, num_classes,
                        imputation_method='missforest', heads=heads, dropout=dropout)


def get_baseline_model(method, in_dim, hidden_dim, out_dim, num_classes, heads=8, dropout=0.6):
    """
    Factory function to get a baseline model.

    Args:
        method: 'mean', 'knn', 'mice', or 'missforest'
        in_dim: input feature dimension
        hidden_dim: GAT hidden dimension
        out_dim: embedding dimension
        num_classes: number of classes
        heads: number of attention heads
        dropout: dropout rate

    Returns:
        model: ImputationBaseline instance
    """
    method = method.lower()

    if method == 'mean':
        return MeanImputationGAT(in_dim, hidden_dim, out_dim, num_classes, heads, dropout)
    elif method == 'knn':
        return KNNImputationGAT(in_dim, hidden_dim, out_dim, num_classes, heads, dropout)
    elif method == 'mice':
        return MICEImputationGAT(in_dim, hidden_dim, out_dim, num_classes, heads, dropout)
    elif method == 'missforest':
        return MissForestImputationGAT(in_dim, hidden_dim, out_dim, num_classes, heads, dropout)
    else:
        raise ValueError(f"Unknown baseline method: {method}")


# Example usage (not executed, just for reference)
if __name__ == '__main__':
    """
    Example usage:

    # Load data
    from datasets import PublicGraphDataset
    from missingness import introduce_structured_missingness

    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    # Introduce missingness
    mask, masked_features = introduce_structured_missingness(
        data.x, data.edge_index, mechanism='mcar', target_ratio=0.5, seed=42
    )

    # Create baseline model
    model = get_baseline_model(
        method='knn',  # or 'mean', 'mice', 'missforest'
        in_dim=dataset.num_features,
        hidden_dim=128,
        out_dim=64,
        num_classes=dataset.num_classes,
        heads=8,
        dropout=0.6
    )

    # Forward pass
    embeddings, logits = model(masked_features, mask, data.edge_index)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    embeddings, logits = model(masked_features, mask, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    """
    pass
