import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import List, Optional, Sequence


class GATEncoder(nn.Module):
    """
    Dynamic multi-layer GATv2 encoder.

    Args:
        in_dim: input feature dimension
        layer_dims: list of output dimensions per layer (final entry is latent size)
        heads: optional list of number of heads per layer (same length as layer_dims); default = 1 for all
        dropout: dropout applied after each non-final layer
        concat_last: if True, final layer concatenates heads (latent dim = layer_dims[-1] * heads[-1])
    """
    def __init__(
        self,
        in_dim: int,
        layer_dims: List[int],
        heads: Optional[List[int]] = None,
        dropout: float = 0.1,
        concat_last: bool = False,
    ):
        super().__init__()
        if len(layer_dims) == 0:
            raise ValueError("layer_dims must contain at least one dimension.")
        if heads is not None and len(heads) != len(layer_dims):
            raise ValueError("heads length must match layer_dims length.")
        self.dropout = dropout
        self.layer_dims = layer_dims
        self.heads = heads or [1] * len(layer_dims)
        self.concat_last = concat_last

        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for i, (out_dim, h) in enumerate(zip(layer_dims, self.heads)):
            is_last = i == len(layer_dims) - 1
            concat = (not is_last) or (is_last and concat_last)
            conv = GATv2Conv(
                prev_dim,
                out_dim,
                heads=h,
                concat=concat,
                dropout=dropout,
            )
            self.layers.append(conv)
            prev_dim = out_dim * h if concat else out_dim

        self.latent_dim = prev_dim  # resulting embedding dimension (after final layer)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            is_last = i == len(self.layers) - 1
            if not is_last:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [num_nodes, latent_dim]

    def encode(self, data):
        return self.forward(data.x, data.edge_index)


class MLPProjector(nn.Module):
    """
    Simple MLP projector for non-graph inputs (e.g., PPR-diffused features).

    Args:
        in_dim: input feature size
        layer_dims: list of hidden dims ending with output (latent) dim
        activation: 'relu'|'gelu'|'elu'|'tanh'|'leaky_relu'|'identity'
        dropout: dropout after each hidden layer
        batch_norm: apply BatchNorm1d after linear (before activation) for hidden layers
    """
    def __init__(
        self,
        in_dim: int,
        layer_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        if len(layer_dims) == 0:
            raise ValueError("layer_dims must contain at least one dimension.")
        act_lookup = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "identity": nn.Identity,
        }
        if activation not in act_lookup:
            raise ValueError(f"Unknown activation: {activation}")
        Act = act_lookup[activation]

        layers = []
        prev = in_dim
        for i, dim in enumerate(layer_dims):
            is_last = i == len(layer_dims) - 1
            layers.append(nn.Linear(prev, dim))
            if not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(Act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev = dim
        self.net = nn.Sequential(*layers)
        self.latent_dim = layer_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [num_nodes, latent_dim]


class PPRGlobalEncoder(nn.Module):
    """
    Global encoder for PPR-diffused features.

    Expected usage:
      - Precompute X_ppr outside (via PPR diffusion on X)
      - Feed X_ppr to this encoder to obtain z_g

    Args:
        in_dim: dimension of X_ppr
        layer_dims: projector dims ending at latent_dim (match local encoders' latent_dim)
        activation, dropout, batch_norm: see MLPProjector
    """
    def __init__(
        self,
        in_dim: int,
        layer_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.projector = MLPProjector(
            in_dim=in_dim,
            layer_dims=layer_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.latent_dim = self.projector.latent_dim

    def forward(self, x_ppr: torch.Tensor) -> torch.Tensor:
        return self.projector(x_ppr)  # [num_nodes, latent_dim]


class AttentionFusion(nn.Module):
    """
    Attention-based fusion over multiple view embeddings {z^(v)}.

    For each node i and view v:
        e_i^(v) = w^T tanh(W z_i^(v))
        alpha_i^(v) = softmax_v(e_i^(v))
        z_i = sum_v alpha_i^(v) * z_i^(v)

    Assumes all views share the same latent_dim.

    Args:
        latent_dim: embedding dimension per view
        num_views: number of views to fuse (e.g., 2 or 3)
    """
    def __init__(self, latent_dim: int, num_views: int):
        super().__init__()
        if num_views < 2:
            raise ValueError("num_views must be >= 2")
        self.num_views = num_views
        self.W = nn.Linear(latent_dim, latent_dim, bias=True)
        self.w = nn.Parameter(torch.randn(latent_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.normal_(self.w, mean=0.0, std=0.02)

    def forward(self, views: Sequence[torch.Tensor]):
        """
        Args:
            views: list/tuple of tensors, each [N, D], length == num_views

        Returns:
            fused: [N, D]
            alpha: [N, V] attention weights per view (useful for analysis)
        """
        if len(views) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views)}.")
        # [N, V, D]
        Z = torch.stack(views, dim=1)
        # Score per view: e = w^T tanh(W z_v)
        # Apply W & tanh per view
        Zw = torch.tanh(self.W(Z))               # [N, V, D]
        e = torch.matmul(Zw, self.w)             # [N, V]
        alpha = torch.softmax(e, dim=1)          # [N, V]
        fused = (alpha.unsqueeze(-1) * Z).sum(1) # [N, D]
        return fused, alpha


class ReliabilityFusion(nn.Module):
    """
    Reliability-aware fusion over multiple view embeddings {z^(v)}.

    Combines two reliability metrics:
    1. Missing rate per node: r_i = 1 - (observed features / total features)
    2. Neighbor consistency per view: u_i^(v) = avg cosine similarity with neighbors

    Fusion weights computed as:
        s_{i,G} = -α * r_i - η * u_i^{(G)}
        s_{i,S} = -α * r_i - η * u_i^{(S)}
        s_{i,P} = -β * r_i - η * u_i^{(P)}
        w_{i,v} = softmax([s_{i,G}, s_{i,S}, s_{i,P}])
        z_i = w_{i,G} * z_i^{(G)} + w_{i,S} * z_i^{(S)} + w_{i,P} * z_i^{(P)}

    Args:
        alpha: Weight for missing rate in geo/sim views
        beta: Weight for missing rate in PPR view
        eta: Weight for neighbor consistency across all views
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def compute_missing_rate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-node missing rate.

        Args:
            x: Feature matrix [N, D] (may contain NaN)

        Returns:
            r: Missing rate per node [N], where r_i ∈ [0, 1]
        """
        observed = (~torch.isnan(x)).float()  # [N, D], 1 = observed, 0 = missing
        r = 1.0 - observed.mean(dim=1)  # [N], missing rate per node
        return r

    def compute_neighbor_consistency(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-node neighbor consistency (average cosine similarity with neighbors).

        Optimized vectorized implementation for sparse graphs using scatter operations.

        Args:
            z: Node embeddings [N, D]
            edge_index: Edge indices [2, E]

        Returns:
            u: Consistency per node [N], higher = more consistent with neighbors
        """
        num_nodes = z.size(0)

        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index_filtered = edge_index[:, mask]

        if edge_index_filtered.size(1) == 0:
            # No edges: return zeros
            return torch.zeros(num_nodes, device=z.device)

        # Normalize embeddings for cosine similarity
        z_norm = F.normalize(z, dim=1)  # [N, D]

        # Get source and target embeddings for each edge
        src_nodes = edge_index_filtered[0]  # [E]
        dst_nodes = edge_index_filtered[1]  # [E]

        src_embs = z_norm[src_nodes]  # [E, D]
        dst_embs = z_norm[dst_nodes]  # [E, D]

        # Compute cosine similarity for each edge (vectorized)
        similarities = (src_embs * dst_embs).sum(dim=1)  # [E]

        # Aggregate: sum similarities per source node
        sim_sum = torch.zeros(num_nodes, device=z.device)
        sim_sum.scatter_add_(0, src_nodes, similarities)

        # Count neighbors per node
        neighbor_count = torch.zeros(num_nodes, device=z.device)
        ones = torch.ones(edge_index_filtered.size(1), device=z.device)
        neighbor_count.scatter_add_(0, src_nodes, ones)

        # Compute average (avoid division by zero)
        u = torch.where(
            neighbor_count > 0,
            sim_sum / neighbor_count,
            torch.zeros_like(sim_sum)
        )

        return u

    def forward(
        self,
        views: Sequence[torch.Tensor],
        x: torch.Tensor,
        edge_index_geo: torch.Tensor,
        edge_index_sim: torch.Tensor,
    ) -> tuple:
        """
        Args:
            views: [z_geo, z_sim, z_ppr] each [N, D]
            x: Original features [N, D] (for computing missing rate)
            edge_index_geo: Geo graph edges [2, E_geo]
            edge_index_sim: Sim graph edges [2, E_sim]

        Returns:
            fused: [N, D]
            weights: [N, 3] fusion weights per node
        """
        if len(views) != 3:
            raise ValueError(f"Expected 3 views, got {len(views)}.")

        z_geo, z_sim, z_ppr = views
        num_nodes = z_geo.size(0)

        # 1. Compute missing rate per node
        r = self.compute_missing_rate(x)  # [N]

        # 2. Compute neighbor consistency per view
        u_geo = self.compute_neighbor_consistency(z_geo, edge_index_geo)  # [N]
        u_sim = self.compute_neighbor_consistency(z_sim, edge_index_sim)  # [N]
        u_ppr = self.compute_neighbor_consistency(z_ppr, edge_index_geo)  # [N] (uses geo edges)

        # 3. Compute reliability scores
        s_geo = -self.alpha * r - self.eta * u_geo  # [N]
        s_sim = -self.alpha * r - self.eta * u_sim  # [N]
        s_ppr = -self.beta * r - self.eta * u_ppr   # [N]

        # Stack scores: [N, 3]
        scores = torch.stack([s_geo, s_sim, s_ppr], dim=1)

        # 4. Softmax to get fusion weights
        weights = torch.softmax(scores, dim=1)  # [N, 3]

        # 5. Weighted fusion
        Z = torch.stack(views, dim=1)  # [N, 3, D]
        fused = (weights.unsqueeze(-1) * Z).sum(dim=1)  # [N, D]

        return fused, weights
