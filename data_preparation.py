import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData


@dataclass
class ClusterPreparationStats:
    cluster_id: int
    num_nodes: int
    num_edges_created: int


@dataclass
class ClusterOverlapInfo:
    cluster_id: int
    global_node_ids: torch.Tensor
    overlap_matrix: torch.Tensor
    x_filled: torch.Tensor  # Features with NaN filled by median
    adjusted_distances: torch.Tensor  # Overlap-aware distances for edge scoring


def standard_scale_features(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard scale features column-wise: (x - mean) / std.
    NaNs are ignored in mean/std computation and preserved in output.
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, num_features]")

    x = x.to(torch.float32)
    observed = ~torch.isnan(x)
    counts = observed.sum(dim=0).to(torch.float32)
    valid_cols = counts > 0

    x_filled = torch.nan_to_num(x, nan=0.0)
    means = torch.zeros(x.size(1), dtype=torch.float32, device=x.device)
    means[valid_cols] = x_filled[:, valid_cols].sum(dim=0) / counts[valid_cols]

    centered = x_filled - means.unsqueeze(0)
    centered[~observed] = 0.0
    var = torch.zeros_like(means)
    var[valid_cols] = (centered[:, valid_cols].pow(2).sum(dim=0) / counts[valid_cols])
    std = torch.sqrt(var + eps)
    std = torch.clamp(std, min=eps)

    x_scaled = (x_filled - means.unsqueeze(0)) / std.unsqueeze(0)
    x_scaled[~observed] = float("nan")
    return x_scaled, means, std


def fill_nan_with_median(x: torch.Tensor) -> torch.Tensor:
    """
    Fill NaN values in feature matrix with median of each feature (column-wise).

    Args:
        x: Node feature tensor [num_nodes, num_features] with possible NaNs.

    Returns:
        x_filled: Copy of x with NaNs replaced by feature-wise median.
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, num_features]")

    x_filled = x.clone()

    for col_idx in range(x.size(1)):
        col = x[:, col_idx]
        mask = ~torch.isnan(col)

        if mask.any():
            # Compute median of non-NaN values
            median_val = col[mask].median()
            # Fill NaN positions with median
            x_filled[~mask, col_idx] = median_val
        else:
            # If entire column is NaN, fill with 0
            x_filled[:, col_idx] = 0.0

    return x_filled


def random_mask_features(
    x: torch.Tensor,
    mask_portion: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a portion of observed entries in X by setting them to NaN.

    Args:
        x: Node feature tensor [num_nodes, num_features].
        mask_portion: Fraction in [0, 1] of currently observed entries to mask.
        seed: Optional random seed for reproducibility.

    Returns:
        x_masked: Copy of x with selected entries set to NaN.
        masked_positions: Bool tensor same shape as x; True where newly masked.
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, num_features]")
    if not (0.0 <= mask_portion <= 1.0):
        raise ValueError("mask_portion must be between 0 and 1")

    observed = ~torch.isnan(x)
    observed_indices = observed.nonzero(as_tuple=False)
    num_observed = observed_indices.size(0)

    if num_observed == 0 or mask_portion == 0.0:
        return x.clone(), torch.zeros_like(observed, dtype=torch.bool)

    num_to_mask = int(mask_portion * num_observed)
    if num_to_mask == 0:
        return x.clone(), torch.zeros_like(observed, dtype=torch.bool)

    if seed is not None:
        generator = torch.Generator(device=observed_indices.device)
        generator.manual_seed(seed)
        perm = torch.randperm(num_observed, generator=generator, device=observed_indices.device)
    else:
        perm = torch.randperm(num_observed, device=observed_indices.device)

    chosen = observed_indices[perm[:num_to_mask]]

    masked_positions = torch.zeros_like(observed, dtype=torch.bool)
    masked_positions[chosen[:, 0], chosen[:, 1]] = True

    x_masked = x.clone()
    x_masked[masked_positions] = float("nan")
    return x_masked, masked_positions


def cluster_graph(
    data: Data,
    target_cluster_size: int = 1000,
    recursive: bool = False,
    log: bool = False,
) -> ClusterData:
    """Partition a graph into clusters using PyG's ClusterData (METIS-based)."""
    if target_cluster_size <= 0:
        raise ValueError("target_cluster_size must be > 0")
    if data.num_nodes is None:
        raise ValueError("Input graph must have a known number of nodes")

    num_parts = max(1, math.ceil(data.num_nodes / target_cluster_size))
    return ClusterData(data, num_parts=num_parts, recursive=recursive, log=log)


def compute_overlap_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute N x N overlap matrix where entry (i, j) is the number of shared
    non-missing features between nodes i and j.
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, num_features]")

    observed = (~torch.isnan(x)).to(torch.float32)
    overlap = observed @ observed.T
    return overlap


def compute_overlap_aware_distance_matrix(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute three N x N matrices for node features in one cluster:
    - overlap: count of shared observed features per node pair
    - distance: Euclidean distance computed only on shared observed features
    - adjusted_distance: distance / overlap (lower is better), inf if overlap == 0
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, num_features]")

    observed = (~torch.isnan(x)).to(torch.float32)
    x_filled = torch.nan_to_num(x, nan=0.0).to(torch.float32)

    # Overlap_{i,j} = sum_k 1(obs_{i,k}) * 1(obs_{j,k})
    overlap = observed @ observed.T

    # Distance on common observed dims only, without constructing [N, N, F].
    x_sq = x_filled * x_filled
    term1 = x_sq @ observed.T
    term2 = observed @ x_sq.T
    cross = x_filled @ x_filled.T

    dist_sq = term1 + term2 - 2.0 * cross
    dist_sq = torch.clamp(dist_sq, min=0.0)
    distance = torch.sqrt(dist_sq + eps)

    adjusted = torch.full_like(distance, float("inf"))
    valid = overlap > 0
    adjusted[valid] = distance[valid] / overlap[valid]

    # Exclude self-neighboring.
    adjusted.fill_diagonal_(float("inf"))

    return overlap, distance, adjusted


def knn_edges_from_adjusted_distances(
    adjusted_distance: torch.Tensor,
    global_node_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Create directed top-k edges (src -> dst) from an adjusted distance matrix."""
    if k <= 0:
        raise ValueError("k must be > 0")

    n = adjusted_distance.size(0)
    edges_src: List[torch.Tensor] = []
    edges_dst: List[torch.Tensor] = []

    for i in range(n):
        row = adjusted_distance[i]
        valid_mask = torch.isfinite(row)
        num_valid = int(valid_mask.sum().item())
        if num_valid == 0:
            continue

        k_eff = min(k, num_valid)
        masked_row = row.clone()
        masked_row[~valid_mask] = float("inf")
        _, nn_idx = torch.topk(masked_row, k=k_eff, largest=False)

        src = torch.full((k_eff,), global_node_ids[i], dtype=torch.long)
        dst = global_node_ids[nn_idx]
        edges_src.append(src)
        edges_dst.append(dst)

    if not edges_src:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.stack([torch.cat(edges_src), torch.cat(edges_dst)], dim=0)
    return edge_index


def build_overlap_aware_knn_graph(
    data: Data,
    k: int = 20,
    target_cluster_size: int = 1000,
    recursive: bool = False,
    log: bool = False,
    make_undirected: bool = True,
) -> Tuple[Data, List[ClusterPreparationStats]]:
    """
    Build a second graph with edges from overlap-aware KNN inside each cluster.

    Workflow:
    1) Partition nodes with ClusterData.
    2) For each cluster, compute overlap-aware pairwise distances.
    3) Keep top-k nearest neighbors per node (using adjusted distance).
    4) Merge all cluster edges into one graph.
    """
    if data.x is None:
        raise ValueError("Input graph must contain node features in data.x")

    clusters = cluster_graph(
        data=data,
        target_cluster_size=target_cluster_size,
        recursive=recursive,
        log=log,
    )

    edge_chunks: List[torch.Tensor] = []
    stats: List[ClusterPreparationStats] = []

    for cluster_id in range(len(clusters)):
        subgraph = clusters[cluster_id]
        if not hasattr(subgraph, "n_id"):
            raise RuntimeError(
                "Cluster subgraph does not provide n_id. "
                "Please ensure your torch-geometric version supports ClusterData n_id mapping."
            )

        global_node_ids = subgraph.n_id.to(torch.long)
        overlap, _, adjusted = compute_overlap_aware_distance_matrix(subgraph.x)
        cluster_edges = knn_edges_from_adjusted_distances(
            adjusted_distance=adjusted,
            global_node_ids=global_node_ids,
            k=k,
        )

        if cluster_edges.numel() > 0:
            edge_chunks.append(cluster_edges)

        stats.append(
            ClusterPreparationStats(
                cluster_id=cluster_id,
                num_nodes=subgraph.num_nodes,
                num_edges_created=cluster_edges.size(1),
            )
        )

    if edge_chunks:
        edge_index = torch.cat(edge_chunks, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Remove duplicate directed edges.
    if edge_index.numel() > 0:
        edge_index = torch.unique(edge_index, dim=1)

    if make_undirected and edge_index.numel() > 0:
        rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, rev], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    second_graph = data.clone()
    second_graph.edge_index = edge_index

    # Avoid stale edge_attr mismatch if original graph had edge attributes.
    if hasattr(second_graph, "edge_attr"):
        second_graph.edge_attr = None

    return second_graph, stats


class GraphPreparationPipeline:
    """
    End-to-end preprocessing pipeline for graph-based imputation:
    1) Standard scale node features (preserving NaNs).
    2) Randomly mask a portion of observed entries.
    3) Cluster graph with PyG ClusterData.
    4) Build overlap-aware top-k KNN similarity graph inside each cluster.
    5) Keep original graph structure (geo_graph).
    6) Return 3 views: geo_graph, sim_graph (KNN), and masked features for PPR.

    The pipeline assumes an existing base graph (e.g., Cora, CiteSeer, PubMed, etc.)
    and creates a similarity graph based on feature overlap.
    """

    def __init__(
        self,
        mask_portion: float = 0.3,
        k: int = 20,
        target_cluster_size: int = 1000,
        recursive: bool = False,
        log: bool = False,
        make_undirected: bool = True,
        mask_seed: Optional[int] = None,
    ):
        self.mask_portion = mask_portion
        self.k = k
        self.target_cluster_size = target_cluster_size
        self.recursive = recursive
        self.log = log
        self.make_undirected = make_undirected
        self.mask_seed = mask_seed

        # Inputs / outputs
        self.input_graph: Optional[Data] = None  # Original geo graph
        self.geo_graph: Optional[Data] = None    # Geo graph with masked features
        self.sim_graph: Optional[Data] = None    # KNN similarity graph

        # Feature copies and scaling stats
        self.x_original: Optional[torch.Tensor] = None
        self.x_scaled: Optional[torch.Tensor] = None
        self.x_masked: Optional[torch.Tensor] = None
        self.x_full: Optional[torch.Tensor] = None  # Full features before masking (for evaluation)
        self.feature_means: Optional[torch.Tensor] = None
        self.feature_stds: Optional[torch.Tensor] = None
        self.masked_positions: Optional[torch.Tensor] = None

        # Clustering / overlap / graph build metadata
        self.clusters: Optional[ClusterData] = None
        self.cluster_stats: List[ClusterPreparationStats] = []
        self.cluster_overlaps: List[ClusterOverlapInfo] = []

    def run(self, data: Data) -> Dict[str, Data]:
        """
        Process the input graph and return 3 views for multi-view learning.

        Args:
            data: Input PyG Data object with x (features) and edge_index (base graph structure)

        Returns:
            Dictionary containing:
                - 'geo_graph': Original graph structure with masked features
                - 'sim_graph': KNN similarity graph with masked features
                - 'x_full': Full scaled features (before masking) for evaluation
                - 'x_masked': Masked features for model input
                - 'masked_positions': Boolean mask indicating which entries were masked
        """
        if data.x is None:
            raise ValueError("Input graph must contain node features in data.x")
        if data.edge_index is None:
            raise ValueError("Input graph must contain edge_index (base graph structure)")

        self.input_graph = data
        self.x_original = data.x.clone()

        # Step 1: Standard scale features
        x_scaled, means, stds = standard_scale_features(data.x)
        self.x_scaled = x_scaled
        self.x_full = x_scaled.clone()  # Keep full features for evaluation
        self.feature_means = means
        self.feature_stds = stds

        # Step 2: Random masking
        x_masked, masked_positions = random_mask_features(
            x=x_scaled,
            mask_portion=self.mask_portion,
            seed=self.mask_seed,
        )
        self.x_masked = x_masked
        self.masked_positions = masked_positions

        # Step 3: Create geo_graph (original structure + masked features)
        self.geo_graph = data.clone()
        self.geo_graph.x = x_masked

        # Step 4: Build similarity graph via clustering + KNN
        # Create temporary data object for clustering
        temp_data = Data(x=x_masked, num_nodes=data.num_nodes)

        self.clusters = cluster_graph(
            data=temp_data,
            target_cluster_size=self.target_cluster_size,
            recursive=self.recursive,
            log=self.log,
        )

        self.cluster_stats = []
        self.cluster_overlaps = []
        edge_chunks: List[torch.Tensor] = []

        for cluster_id in range(len(self.clusters)):
            subgraph = self.clusters[cluster_id]
            if not hasattr(subgraph, "n_id"):
                raise RuntimeError(
                    "Cluster subgraph does not provide n_id. "
                    "Please ensure your torch-geometric version supports ClusterData n_id mapping."
                )

            global_node_ids = subgraph.n_id.to(torch.long)

            # Fill NaN values with median for this cluster
            x_filled_cluster = fill_nan_with_median(subgraph.x)

            overlap, _, adjusted = compute_overlap_aware_distance_matrix(subgraph.x)
            cluster_edges = knn_edges_from_adjusted_distances(
                adjusted_distance=adjusted,
                global_node_ids=global_node_ids,
                k=self.k,
            )

            if cluster_edges.numel() > 0:
                edge_chunks.append(cluster_edges)

            self.cluster_overlaps.append(
                ClusterOverlapInfo(
                    cluster_id=cluster_id,
                    global_node_ids=global_node_ids,
                    overlap_matrix=overlap,
                    x_filled=x_filled_cluster,
                    adjusted_distances=adjusted,
                )
            )
            self.cluster_stats.append(
                ClusterPreparationStats(
                    cluster_id=cluster_id,
                    num_nodes=subgraph.num_nodes,
                    num_edges_created=cluster_edges.size(1),
                )
            )

        # Step 5: Assemble KNN edges
        if edge_chunks:
            edge_index = torch.cat(edge_chunks, dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if self.make_undirected and edge_index.numel() > 0:
            rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            edge_index = torch.unique(edge_index, dim=1)

        # Step 6: Create sim_graph (KNN structure + masked features)
        self.sim_graph = Data(x=x_masked, edge_index=edge_index, num_nodes=data.num_nodes)

        # Step 7: Create filled versions (NaN replaced with median) for training
        x_masked_filled = fill_nan_with_median(x_masked)
        self.geo_graph_filled = data.clone()
        self.geo_graph_filled.x = x_masked_filled
        self.sim_graph_filled = Data(x=x_masked_filled, edge_index=edge_index, num_nodes=data.num_nodes)

        # Step 8: Build initial edge scores (for graph update during training)
        initial_edge_scores = self._build_initial_edge_scores()

        return {
            'geo_graph': self.geo_graph,
            'sim_graph': self.sim_graph,
            'geo_graph_filled': self.geo_graph_filled,
            'sim_graph_filled': self.sim_graph_filled,
            'x_full': self.x_full,
            'x_masked': self.x_masked,
            'x_masked_filled': x_masked_filled,
            'masked_positions': self.masked_positions,
            'initial_edge_scores': initial_edge_scores,
            'cluster_overlaps': self.cluster_overlaps,
        }

    def _build_initial_edge_scores(self) -> Dict[Tuple[int, int], float]:
        """
        Build a dictionary mapping each edge (src, dst) to its initial distance score.
        Lower score = better edge (closer in feature space).

        Returns:
            Dict mapping (src_node, dst_node) -> distance_score
        """
        edge_scores = {}

        for overlap_info in self.cluster_overlaps:
            global_node_ids = overlap_info.global_node_ids.cpu().numpy()
            adjusted_dist = overlap_info.adjusted_distances.cpu()

            # For each node in this cluster
            for local_i, global_i in enumerate(global_node_ids):
                # Get distances from this node to all others in cluster
                distances = adjusted_dist[local_i]

                # Sort to get top-k nearest neighbors
                valid_mask = torch.isfinite(distances)
                if not valid_mask.any():
                    continue

                valid_distances = distances.clone()
                valid_distances[~valid_mask] = float('inf')

                # Get indices of k nearest neighbors
                k_eff = min(self.k, valid_mask.sum().item())
                if k_eff == 0:
                    continue

                _, local_neighbors = torch.topk(valid_distances, k=k_eff, largest=False)

                # Store edge scores
                for local_j in local_neighbors:
                    global_j = global_node_ids[local_j]
                    if global_i != global_j:
                        distance = distances[local_j].item()
                        if torch.isfinite(torch.tensor(distance)):
                            # Store as (src, dst) -> distance
                            # Use negative distance so higher score = better
                            edge_scores[(int(global_i), int(global_j))] = -distance

        return edge_scores


if __name__ == "__main__":
    # Example usage with Cora dataset:
    from torch_geometric.datasets import Planetoid

    # Load a dataset (e.g., Cora, CiteSeer, PubMed)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # Initialize pipeline
    pipeline = GraphPreparationPipeline(
        mask_portion=0.3,
        k=20,
        target_cluster_size=1000,
        mask_seed=42
    )

    # Run pipeline to get 3 views
    result = pipeline.run(data)

    print("=" * 60)
    print("Graph Preparation Pipeline Results")
    print("=" * 60)
    print(f"Original graph nodes: {data.num_nodes}, edges: {data.edge_index.size(1)}")
    print(f"Geo graph edges: {result['geo_graph'].edge_index.size(1)}")
    print(f"Sim graph edges: {result['sim_graph'].edge_index.size(1)}")
    print(f"Masked positions: {result['masked_positions'].sum().item()} / {result['masked_positions'].numel()}")
    print(f"Mask ratio: {result['masked_positions'].sum().item() / result['masked_positions'].numel():.3f}")
    print(f"\nCluster statistics:")
    for stat in pipeline.cluster_stats[:5]:
        print(f"  Cluster {stat.cluster_id}: {stat.num_nodes} nodes, {stat.num_edges_created} edges")
    print("=" * 60)
