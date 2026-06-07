import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SharedGATEncoder(nn.Module):
    """
    Shared GAT encoder for both base and PPR views.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, heads=8, dropout=0.6):
        """
        Args:
            in_dim: input dimension
            hidden_dim: hidden dimension per head
            out_dim: output embedding dimension
            num_layers: number of GAT layers
            heads: number of attention heads
            dropout: dropout probability
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim  # Store for later reference

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Last layer (single head for output)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        else:
            # If only 1 layer, output directly
            self.convs[0] = GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x: [n_nodes, in_dim] input features
            edge_index: [2, n_edges] graph edges

        Returns:
            z: [n_nodes, out_dim] node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class ReliabilityWeightedBarlowTwins(nn.Module):
    """
    Reliability-weighted Barlow Twins loss for contrastive learning.
    """

    def __init__(self, lambda_bt=0.005):
        """
        Args:
            lambda_bt: weight for off-diagonal redundancy penalty
        """
        super().__init__()
        self.lambda_bt = lambda_bt

    def forward(self, z_base, z_ppr, reliability, eps=1e-8):
        """
        Compute reliability-weighted Barlow Twins loss.

        Args:
            z_base: [batch_size, embed_dim] base view embeddings
            z_ppr: [batch_size, embed_dim] PPR view embeddings
            reliability: [batch_size] node reliability scores
            eps: numerical stability constant

        Returns:
            loss: scalar loss value
        """
        batch_size, embed_dim = z_base.shape

        # Normalize embeddings (zero mean, unit variance per dimension)
        z_base_norm = (z_base - z_base.mean(dim=0)) / (z_base.std(dim=0) + eps)
        z_ppr_norm = (z_ppr - z_ppr.mean(dim=0)) / (z_ppr.std(dim=0) + eps)

        # Compute reliability-weighted cross-correlation matrix
        reliability = reliability.view(-1, 1)  # [batch_size, 1]
        reliability_sum = reliability.sum() + eps

        # C[a,b] = sum_i (r_i * z_base[i,a] * z_ppr[i,b]) / sum_i r_i
        C = (z_base_norm.T @ (reliability * z_ppr_norm)) / reliability_sum  # [embed_dim, embed_dim]

        # Diagonal loss: encourage agreement between views
        diagonal_loss = ((1 - torch.diagonal(C)) ** 2).sum()

        # Off-diagonal loss: discourage redundant dimensions
        off_diagonal_mask = ~torch.eye(embed_dim, dtype=bool, device=C.device)
        off_diagonal_loss = (C[off_diagonal_mask] ** 2).sum()

        # Total loss
        loss = diagonal_loss + self.lambda_bt * off_diagonal_loss

        return loss


class ContrastiveGraphLearning(nn.Module):
    """
    Main model: Contrastive Graph Learning with Reliability-Aware Fusion.
    """

    def __init__(self, in_dim, hidden_dim, embed_dim, num_classes,
                 num_layers=2, heads=8, dropout=0.6, lambda_bt=0.005):
        """
        Args:
            in_dim: input feature dimension (after missingness encoding: 3 * n_features)
            hidden_dim: hidden dimension for GAT
            embed_dim: embedding dimension
            num_classes: number of output classes
            num_layers: number of GAT layers
            heads: number of attention heads
            dropout: dropout probability
            lambda_bt: Barlow Twins off-diagonal penalty
        """
        super().__init__()

        # Shared encoder for both views
        self.encoder = SharedGATEncoder(in_dim, hidden_dim, embed_dim,
                                       num_layers=num_layers, heads=heads, dropout=dropout)

        # Reliability-weighted Barlow Twins loss
        self.barlow_twins = ReliabilityWeightedBarlowTwins(lambda_bt=lambda_bt)

        # Learnable reliability combination parameter
        self.rho = nn.Parameter(torch.tensor(0.0))

        # Task-specific head (classification)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Optional reconstruction decoder
        self.use_reconstruction = False
        self.decoder = None

    def enable_reconstruction(self, out_dim):
        """Enable optional reconstruction objective."""
        self.use_reconstruction = True
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, h0, edge_index_base, edge_index_ppr):
        """
        Forward pass through both graph views.

        Args:
            h0: [n_nodes, in_dim] missingness-encoded input features
            edge_index_base: [2, n_edges_base] base graph edges
            edge_index_ppr: [2, n_edges_ppr] PPR graph edges

        Returns:
            z_base: [n_nodes, embed_dim] base view embeddings
            z_ppr: [n_nodes, embed_dim] PPR view embeddings
            logits: [n_nodes, num_classes] classification logits
        """
        # Encode both views with shared encoder
        z_base = self.encoder(h0, edge_index_base)
        z_ppr = self.encoder(h0, edge_index_ppr)

        # Classification from base view
        logits = self.classifier(z_base)

        return z_base, z_ppr, logits

    def compute_reliability(self, r_self, r_nbr):
        """
        Compute final reliability score with learned parameter.

        Args:
            r_self: [n_nodes] self-observation reliability
            r_nbr: [n_nodes] neighborhood reliability

        Returns:
            r: [n_nodes] final reliability scores
        """
        weight = torch.sigmoid(self.rho)
        r = weight * r_self + (1 - weight) * r_nbr
        return r

    def contrastive_loss(self, z_base, z_ppr, reliability):
        """
        Compute reliability-weighted Barlow Twins loss.

        Args:
            z_base: [n_nodes, embed_dim] base view embeddings
            z_ppr: [n_nodes, embed_dim] PPR view embeddings
            reliability: [n_nodes] node reliability scores

        Returns:
            loss: scalar contrastive loss
        """
        return self.barlow_twins(z_base, z_ppr, reliability)

    def reconstruction_loss(self, z_base, features_original, mask, eps=1e-8):
        """
        Optional reconstruction loss on observed entries.

        Args:
            z_base: [n_nodes, embed_dim] base view embeddings
            features_original: [n_nodes, n_features] original features
            mask: [n_nodes, n_features] observation mask

        Returns:
            loss: scalar reconstruction loss
        """
        if not self.use_reconstruction or self.decoder is None:
            return torch.tensor(0.0, device=z_base.device)

        # Decode embeddings
        features_recon = self.decoder(z_base)  # [n_nodes, n_features]

        # Compute MSE only on observed entries
        diff = (features_recon - features_original) ** 2
        masked_diff = diff * mask

        # Average over observed entries per node
        node_loss = masked_diff.sum(dim=1) / (mask.sum(dim=1) + eps)
        loss = node_loss.mean()

        return loss


class MissingGraphModel(nn.Module):
    """
    Complete end-to-end model including initialization and encoding.
    """

    def __init__(self, n_features, hidden_dim, embed_dim, num_classes,
                 num_layers=2, heads=8, dropout=0.6, lambda_bt=0.005):
        """
        Args:
            n_features: number of raw node features
            hidden_dim: hidden dimension for GAT
            embed_dim: embedding dimension
            num_classes: number of output classes
            num_layers: number of GAT layers
            heads: number of attention heads
            dropout: dropout probability
            lambda_bt: Barlow Twins penalty weight
        """
        super().__init__()

        # Missingness pattern encoder: [features, mask, availability] -> hidden_dim
        in_dim = 3 * n_features
        self.pattern_encoder = nn.Sequential(
            nn.Linear(in_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        # Main contrastive learning model
        self.model = ContrastiveGraphLearning(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            lambda_bt=lambda_bt
        )

        self.n_features = n_features

    def encode_missingness(self, features, mask):
        """
        Encode missingness pattern.

        Args:
            features: [n_nodes, n_features] (imputed) features
            mask: [n_nodes, n_features] observation mask

        Returns:
            h0: [n_nodes, hidden_dim] encoded input
        """
        # Global feature frequencies
        q = mask.float().mean(dim=0)  # [n_features]

        # Feature-availability pattern
        s = mask * q.unsqueeze(0)  # [n_nodes, n_features]

        # Concatenate
        concatenated = torch.cat([features, mask.float(), s], dim=1)

        # Encode
        h0 = self.pattern_encoder(concatenated)

        return h0

    def forward(self, features, mask, edge_index_base, edge_index_ppr, r_self, r_nbr):
        """
        Full forward pass.

        Args:
            features: [n_nodes, n_features] (imputed) features
            mask: [n_nodes, n_features] observation mask
            edge_index_base: [2, n_edges] base graph
            edge_index_ppr: [2, n_edges_ppr] PPR graph
            r_self: [n_nodes] self reliability
            r_nbr: [n_nodes] neighborhood reliability

        Returns:
            z_base, z_ppr, logits, reliability
        """
        # Encode missingness pattern
        h0 = self.encode_missingness(features, mask)

        # Forward through model
        z_base, z_ppr, logits = self.model(h0, edge_index_base, edge_index_ppr)

        # Compute reliability
        reliability = self.model.compute_reliability(r_self, r_nbr)

        return z_base, z_ppr, logits, reliability


if __name__ == '__main__':
    from datasets import PublicGraphDataset, introduce_missingness
    from initialization import mask_aware_knn_imputation, compute_node_reliability
    from ppr_view import ppr_topk_graph

    print("Testing model components...")

    # Load dataset
    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    # Introduce missingness
    masked_features, mask = introduce_missingness(data.x, mask_ratio=0.3, mechanism='mcar', seed=42)

    # Impute
    imputed_features = mask_aware_knn_imputation(masked_features, mask, data.edge_index, k=5)

    # Compute reliability
    r_self, r_nbr, _ = compute_node_reliability(mask, data.edge_index, k=5)

    # Create PPR view
    ppr_edge_index = ppr_topk_graph(data.edge_index, data.num_nodes, k=10, alpha=0.15)

    print(f"Dataset: {data.num_nodes} nodes, {dataset.num_features} features, {dataset.num_classes} classes")

    # Initialize model
    model = MissingGraphModel(
        n_features=dataset.num_features,
        hidden_dim=128,
        embed_dim=64,
        num_classes=dataset.num_classes,
        num_layers=2,
        heads=8,
        dropout=0.6
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        z_base, z_ppr, logits, reliability = model(
            imputed_features, mask,
            data.edge_index, ppr_edge_index,
            r_self, r_nbr
        )

    print(f"\nForward pass outputs:")
    print(f"  Base embeddings: {z_base.shape}")
    print(f"  PPR embeddings: {z_ppr.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Reliability: {reliability.shape}, mean={reliability.mean():.3f}")

    # Test contrastive loss
    loss_bt = model.model.contrastive_loss(z_base, z_ppr, reliability)
    print(f"\nBarlow Twins loss: {loss_bt.item():.4f}")
