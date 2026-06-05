import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.nn import APPNP
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from typing import Dict, Optional

from data_preparation import GraphPreparationPipeline
from gnn_model import GATEncoder, PPRGlobalEncoder, AttentionFusion, ReliabilityFusion
from decoder_model import MLPDecoder


class MultiViewImputationModel:
    """
    Multi-view graph-based imputation model with 3 views:
    1. Geo view: Original graph structure (e.g., citation network)
    2. Sim view: KNN similarity graph based on feature overlap
    3. Global view: PPR-diffused features (APPNP)

    The model uses contrastive learning (InfoNCE + Barlow Twins) to align
    views and reconstructs masked features via a shared decoder.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        latent_dim: int,
        decoder_dims: list,
        lr: float = 0.001,
        tau: float = 0.5,
        beta_info: float = 0.01,
        beta_barlow: float = 0.01,
        ppr_alpha: float = 0.2,
        ppr_k: int = 10,
        batch_size: int = 512,
        num_parts: int = 10,
        loader_type: str = 'cluster',
        update_graph_every: int = 10,
        warmup_epochs: int = 30,
        blend_alpha: float = 0.7,
        fusion_type: str = 'attention',
        reliability_alpha: float = 1.0,
        reliability_beta: float = 1.0,
        reliability_eta: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dims: List of hidden dimensions for GAT encoders
            latent_dim: Latent embedding dimension
            decoder_dims: List of hidden dimensions for decoder
            lr: Learning rate
            tau: Temperature for InfoNCE loss
            beta_info: Weight for InfoNCE loss
            beta_barlow: Weight for Barlow Twins loss
            ppr_alpha: Teleport probability for PPR
            ppr_k: Number of propagation steps for APPNP
            batch_size: Batch size for neighbor sampling (only used if loader_type='neighbor')
            num_parts: Number of clusters/partitions (only used if loader_type='cluster')
            loader_type: Type of data loader ('cluster' or 'neighbor')
            update_graph_every: Update sim graph every N epochs (0 = no updates)
            warmup_epochs: Number of epochs before starting graph updates
            blend_alpha: Ratio of old edges to keep when updating (0.7 = 70% old, 30% new)
            fusion_type: Type of fusion ('attention' or 'reliability')
            reliability_alpha: Weight for missing rate in geo/sim views (only for reliability fusion)
            reliability_beta: Weight for missing rate in PPR view (only for reliability fusion)
            reliability_eta: Weight for neighbor consistency (only for reliability fusion)
            device: Device to use (cuda/cpu)
        """
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.tau = tau
        self.beta_info = beta_info
        self.beta_barlow = beta_barlow
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.loader_type = loader_type
        self.update_graph_every = update_graph_every
        self.warmup_epochs = warmup_epochs
        self.blend_alpha = blend_alpha
        self.k = ppr_k  # Store k for graph updates
        self.fusion_type = fusion_type

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # PPR diffusion layer (for global view)
        self.ppr = APPNP(K=ppr_k, alpha=ppr_alpha)

        # Encoders (two local + one global)
        self.encoder_geo = GATEncoder(
            in_dim=in_dim,
            layer_dims=hidden_dims + [latent_dim],
            dropout=0.1
        ).to(self.device)

        self.encoder_sim = GATEncoder(
            in_dim=in_dim,
            layer_dims=hidden_dims + [latent_dim],
            dropout=0.1
        ).to(self.device)

        self.encoder_global = PPRGlobalEncoder(
            in_dim=in_dim,
            layer_dims=[latent_dim],
            dropout=0.0
        ).to(self.device)

        # Fusion layer (attention or reliability-based)
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(
                latent_dim=latent_dim,
                num_views=3
            ).to(self.device)
        elif fusion_type == 'reliability':
            self.fusion = ReliabilityFusion(
                alpha=reliability_alpha,
                beta=reliability_beta,
                eta=reliability_eta,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'attention' or 'reliability'.")

        # Decoder (shared)
        self.decoder = MLPDecoder(
            in_dim=latent_dim,
            layer_dims=decoder_dims + [in_dim],
            dropout=0.1
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder_geo.parameters())
            + list(self.encoder_sim.parameters())
            + list(self.encoder_global.parameters())
            + list(self.fusion.parameters())
            + list(self.decoder.parameters()),
            lr=lr,
        )

        # Loss criterion
        self.criterion = nn.L1Loss()

        # Data storage
        self.geo_graph = None
        self.sim_graph = None
        self.x_ppr = None
        self.x_full = None
        self.x_masked_with_nan = None  # Masked features with NaN (for reliability fusion)
        self.masked_positions = None
        self.initial_edge_scores = None  # Initial KNN distances for edge scoring

    def compute_ppr_features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute PPR-diffused features using APPNP."""
        return self.ppr(x, edge_index)

    def blend_edge_indices_per_node(
        self,
        old_edges: torch.Tensor,
        new_edges: torch.Tensor,
        z_fused: torch.Tensor,
        alpha: float = 0.7,
        k: int = 20
    ) -> torch.Tensor:
        """
        For each node, keep alpha*k best old neighbors + (1-alpha)*k best new neighbors.
        Uses similarity scores from both initial distances and latent embeddings.

        Args:
            old_edges: [2, E_old] current edge indices
            new_edges: [2, E_new] proposed edges from latent space KNN
            z_fused: [N, D] fused embeddings for computing latent similarity
            alpha: fraction of old edges to keep (0.7 = 70% old, 30% new)
            k: target number of neighbors per node

        Returns:
            [2, E_blended] blended edge indices
        """
        num_nodes = max(old_edges.max().item(), new_edges.max().item()) + 1
        z_norm = F.normalize(z_fused, dim=1)

        # Build adjacency lists with scores for old edges
        old_adj = {i: [] for i in range(num_nodes)}  # {node_id: [(neighbor, score), ...]}

        for i in range(old_edges.size(1)):
            src, dst = old_edges[0, i].item(), old_edges[1, i].item()

            # Use initial edge score if available, otherwise use latent similarity
            edge_key = (src, dst)
            if edge_key in self.initial_edge_scores:
                score = self.initial_edge_scores[edge_key]
            else:
                # Fallback to latent similarity
                score = (z_norm[src] * z_norm[dst]).sum().item()

            old_adj[src].append((dst, score))

        # Build adjacency lists with scores for new edges (use latent similarity)
        new_adj = {i: [] for i in range(num_nodes)}

        for i in range(new_edges.size(1)):
            src, dst = new_edges[0, i].item(), new_edges[1, i].item()
            score = (z_norm[src] * z_norm[dst]).sum().item()
            new_adj[src].append((dst, score))

        # Sort by score (higher = better)
        for node_id in range(num_nodes):
            old_adj[node_id].sort(key=lambda x: x[1], reverse=True)
            new_adj[node_id].sort(key=lambda x: x[1], reverse=True)

        # Blend per node
        blended_edges = []
        num_old_keep = int(k * alpha)
        num_new_add = k - num_old_keep

        for node_id in range(num_nodes):
            # Keep best old neighbors
            best_old = [n for n, _ in old_adj[node_id][:num_old_keep]]

            # Add best new neighbors (not already in old)
            kept_set = set(best_old)
            best_new = [n for n, _ in new_adj[node_id] if n not in kept_set][:num_new_add]

            # Combine
            final_neighbors = best_old + best_new

            for neighbor in final_neighbors:
                blended_edges.append([node_id, neighbor])

        # Convert to tensor
        if blended_edges:
            result = torch.tensor(blended_edges, dtype=torch.long, device=old_edges.device).t()
            # Make undirected
            result = torch.cat([result, result.flip(0)], dim=1)
            result = torch.unique(result, dim=1)
        else:
            result = old_edges

        return result

    def update_sim_graph(self, epoch: int):
        """
        Update similarity graph based on learned embeddings.
        Keeps best edges from both initial feature space and learned latent space.

        Args:
            epoch: Current training epoch
        """
        # Check if we should update
        if self.update_graph_every == 0:
            return  # Updates disabled

        if epoch < self.warmup_epochs:
            return  # Still in warmup period

        if epoch % self.update_graph_every != 0:
            return  # Not an update epoch

        print(f"\n→ Updating sim graph at epoch {epoch}...")

        with torch.no_grad():
            # Get embeddings
            x = self.sim_graph.x.to(self.device)
            edge_geo = self.geo_graph.edge_index.to(self.device)
            edge_sim = self.sim_graph.edge_index.to(self.device)
            x_ppr = self.x_ppr.to(self.device)

            z_geo = self.encoder_geo(x, edge_geo)
            z_sim = self.encoder_sim(x, edge_sim)
            z_global = self.encoder_global(x_ppr)

            # Fuse (attention or reliability-based)
            if self.fusion_type == 'attention':
                z_fused, _ = self.fusion([z_geo, z_sim, z_global])
            else:  # reliability fusion
                x_masked = self.x_masked_with_nan.to(self.device)
                z_fused, _ = self.fusion(
                    [z_geo, z_sim, z_global],
                    x_masked,
                    edge_geo,
                    edge_sim
                )

            # Compute new KNN graph from latent space
            z_norm = F.normalize(z_fused, dim=1)
            similarity = torch.mm(z_norm, z_norm.T)
            similarity.fill_diagonal_(-1)  # Exclude self-loops

            # Get top-k neighbors per node
            _, indices = torch.topk(similarity, k=self.k, dim=1, largest=True)

            row = torch.arange(indices.size(0), device=self.device).repeat_interleave(self.k)
            col = indices.flatten()
            new_edges = torch.stack([row, col], dim=0)

            # Blend with old edges using per-node strategy
            old_edges = self.sim_graph.edge_index.to(self.device)
            blended_edges = self.blend_edge_indices_per_node(
                old_edges, new_edges, z_fused,
                alpha=self.blend_alpha,
                k=self.k
            )

            # Update sim_graph
            self.sim_graph.edge_index = blended_edges.cpu()

            # Log statistics
            num_old = old_edges.size(1)
            num_new = blended_edges.size(1)

            old_set = set(map(tuple, old_edges.cpu().t().tolist()))
            new_set = set(map(tuple, blended_edges.cpu().t().tolist()))
            num_added = len(new_set - old_set)
            num_removed = len(old_set - new_set)
            num_kept = len(old_set & new_set)

            print(f"  Edges: {num_old} → {num_new}")
            print(f"  Kept: {num_kept}, Added: {num_added}, Removed: {num_removed}")
            print(f"  Change ratio: {(num_added + num_removed) / num_old:.2%}")

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss between two views.
        Encourages agreement between different graph views.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)

        # Concatenate both views
        reps = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.matmul(reps, reps.T) / self.tau

        # Mask diagonal
        mask = torch.eye(2 * N, device=z1.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)

        # Positive pairs: z1[i] <-> z2[i]
        labels = torch.arange(N, device=z1.device)
        labels = torch.cat([labels + N, labels], dim=0)

        return F.cross_entropy(sim, labels)

    def barlow_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, lambda_offdiag: float = 5e-3
    ) -> torch.Tensor:
        """
        Barlow Twins loss for redundancy reduction.
        Encourages decorrelation of embedding dimensions.
        """
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)

        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / z1.size(0)

        # On-diagonal: should be 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal: should be 0
        off_diag = c.fill_diagonal_(0).pow_(2).sum()

        return on_diag + lambda_offdiag * off_diag

    def prepare_data(self, result: Dict):
        """
        Prepare data from pipeline output.

        Args:
            result: Dictionary from GraphPreparationPipeline.run()
        """
        # Use filled graphs for training (NaN replaced with median)
        self.geo_graph = result['geo_graph_filled']
        self.sim_graph = result['sim_graph_filled']
        self.x_full = result['x_full']
        self.masked_positions = result['masked_positions']
        self.initial_edge_scores = result.get('initial_edge_scores', {})

        # Store masked features with NaN (for reliability fusion)
        self.x_masked_with_nan = result['geo_graph'].x  # Original masked features with NaN

        # Compute PPR features on filled geo graph
        print("Computing PPR features on filled graph...")
        self.x_ppr = self.compute_ppr_features(
            self.geo_graph.x,
            self.geo_graph.edge_index
        )
        print(f"PPR features computed: {self.x_ppr.shape}")
        print(f"Initial edge scores loaded: {len(self.initial_edge_scores)} edges")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using mini-batch sampling."""
        self.encoder_geo.train()
        self.encoder_sim.train()
        self.encoder_global.train()
        self.fusion.train()
        self.decoder.train()

        # Create data loaders based on loader_type
        if self.loader_type == 'cluster':
            # Use ClusterLoader for training
            cluster_data_geo = ClusterData(
                self.geo_graph,
                num_parts=self.num_parts,
                recursive=False,
                log=False
            )
            cluster_data_sim = ClusterData(
                self.sim_graph,
                num_parts=self.num_parts,
                recursive=False,
                log=False
            )
            loader_geo = ClusterLoader(cluster_data_geo, batch_size=1, shuffle=True)
            loader_sim = ClusterLoader(cluster_data_sim, batch_size=1, shuffle=True)
        else:  # loader_type == 'neighbor'
            # Use NeighborLoader for training
            loader_geo = NeighborLoader(
                self.geo_graph,
                num_neighbors=[15, 10],
                batch_size=self.batch_size,
                shuffle=True,
            )
            loader_sim = NeighborLoader(
                self.sim_graph,
                num_neighbors=[15, 10],
                batch_size=self.batch_size,
                shuffle=True,
            )

        total_loss = 0
        total_info = 0
        total_barlow = 0
        total_recon = 0
        num_batches = 0

        for batch_geo, batch_sim in zip(loader_geo, loader_sim):
            # Move to device
            x_geo = batch_geo.x.to(self.device)
            x_sim = batch_sim.x.to(self.device)
            edge_geo = batch_geo.edge_index.to(self.device)
            edge_sim = batch_sim.edge_index.to(self.device)

            # Handle different loader types
            if self.loader_type == 'cluster':
                # ClusterLoader: use n_id to get original node indices
                batch_nodes_geo = batch_geo.n_id.to(self.device)
                batch_size_geo = batch_geo.num_nodes
                batch_size_sim = batch_sim.num_nodes
            else:  # NeighborLoader
                # NeighborLoader: seed nodes are first batch_size nodes
                batch_nodes_geo = batch_geo.n_id[: batch_geo.batch_size]
                batch_size_geo = batch_geo.batch_size
                batch_size_sim = batch_sim.batch_size

            # PPR features for this batch
            x_ppr_batch = self.x_ppr[batch_nodes_geo].to(self.device)

            # Encode each view
            z_geo = self.encoder_geo(x_geo, edge_geo)
            z_sim = self.encoder_sim(x_sim, edge_sim)
            z_global = self.encoder_global(x_ppr_batch)

            # Keep only seed nodes (for NeighborLoader) or all nodes (for ClusterLoader)
            if self.loader_type == 'neighbor':
                z_geo = z_geo[: batch_size_geo]
                z_sim = z_sim[: batch_size_sim]
            # For ClusterLoader, use all nodes in the cluster (no slicing needed)

            # Contrastive losses (between geo and sim views)
            loss_info = self.contrastive_loss(z_geo, z_sim)
            loss_barlow = self.barlow_loss(z_geo, z_sim)

            # Fuse 3 views (attention or reliability-based)
            if self.fusion_type == 'attention':
                fused, alpha = self.fusion([z_geo, z_sim, z_global])
            else:  # reliability fusion
                # Get masked features with NaN for this batch
                x_masked_batch = self.x_masked_with_nan[batch_nodes_geo].to(self.device)
                fused, weights = self.fusion(
                    [z_geo, z_sim, z_global],
                    x_masked_batch,
                    edge_geo,
                    edge_sim
                )

            # Decode
            reconstructed = self.decoder(fused)

            # Reconstruction loss (only on masked positions)
            x_target = self.x_full[batch_nodes_geo].to(self.device)
            mask = self.masked_positions[batch_nodes_geo].to(self.device)

            if mask.any():
                loss_recon = self.criterion(reconstructed[mask], x_target[mask])
            else:
                loss_recon = torch.tensor(0.0, device=self.device)

            # Combined loss
            loss = (
                self.beta_info * loss_info
                + self.beta_barlow * loss_barlow
                + loss_recon
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_info += loss_info.item()
            total_barlow += loss_barlow.item()
            total_recon += loss_recon.item()
            num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'info_loss': total_info / num_batches,
            'barlow_loss': total_barlow / num_batches,
            'recon_loss': total_recon / num_batches,
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on masked positions (validation/test)."""
        self.encoder_geo.eval()
        self.encoder_sim.eval()
        self.encoder_global.eval()
        self.fusion.eval()
        self.decoder.eval()

        with torch.no_grad():
            # Full graph forward pass
            x_geo = self.geo_graph.x.to(self.device)
            x_sim = self.sim_graph.x.to(self.device)
            edge_geo = self.geo_graph.edge_index.to(self.device)
            edge_sim = self.sim_graph.edge_index.to(self.device)
            x_ppr = self.x_ppr.to(self.device)

            # Encode
            z_geo = self.encoder_geo(x_geo, edge_geo)
            z_sim = self.encoder_sim(x_sim, edge_sim)
            z_global = self.encoder_global(x_ppr)

            # Fuse (attention or reliability-based)
            if self.fusion_type == 'attention':
                fused, alpha = self.fusion([z_geo, z_sim, z_global])
            else:  # reliability fusion
                x_masked = self.x_masked_with_nan.to(self.device)
                fused, weights = self.fusion(
                    [z_geo, z_sim, z_global],
                    x_masked,
                    edge_geo,
                    edge_sim
                )

            # Decode
            reconstructed = self.decoder(fused)

            # Compute metrics on masked positions
            x_target = self.x_full.to(self.device)
            mask = self.masked_positions.to(self.device)

            mae = F.l1_loss(reconstructed[mask], x_target[mask])
            mse = F.mse_loss(reconstructed[mask], x_target[mask])

            return {
                'mae': mae.item(),
                'mse': mse.item(),
                'rmse': torch.sqrt(mse).item(),
            }

    def train(self, epochs: int, eval_every: int = 10):
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of training epochs
            eval_every: Evaluate every N epochs
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        if self.update_graph_every > 0:
            print(f"Graph updates: Every {self.update_graph_every} epochs after {self.warmup_epochs} warmup")
            print(f"Blend ratio: {self.blend_alpha:.1%} old, {1-self.blend_alpha:.1%} new")
        else:
            print("Graph updates: Disabled")
        print("=" * 60)

        best_mae = float('inf')

        for epoch in range(1, epochs + 1):
            # Update sim graph if needed (before training epoch)
            self.update_sim_graph(epoch)

            train_metrics = self.train_epoch()

            if epoch % eval_every == 0 or epoch == 1:
                eval_metrics = self.evaluate()

                print(
                    f"Epoch {epoch:04d} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"InfoNCE: {self.beta_info * train_metrics['info_loss']:.4f} | "
                    f"Barlow: {self.beta_barlow * train_metrics['barlow_loss']:.4f} | "
                    f"Recon: {train_metrics['recon_loss']:.4f} | "
                    f"Eval MAE: {eval_metrics['mae']:.4f}"
                )

                if eval_metrics['mae'] < best_mae:
                    best_mae = eval_metrics['mae']
            else:
                print(
                    f"Epoch {epoch:04d} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Recon: {train_metrics['recon_loss']:.4f}"
                )

        print("=" * 60)
        print(f"Training complete! Best MAE: {best_mae:.4f}")
        print("=" * 60)

        return best_mae


class ModelTrainer:
    """
    High-level trainer that combines data preparation and model training.
    """

    def __init__(
        self,
        dataset_name: str = 'Cora',
        dataset_root: str = '/tmp',
        mask_portion: float = 0.3,
        k: int = 20,
        target_cluster_size: int = 500,
        hidden_dims: list = [32, 64],
        latent_dim: int = 128,
        decoder_dims: list = [64, 32],
        lr: float = 0.001,
        tau: float = 0.5,
        beta_info: float = 0.01,
        beta_barlow: float = 0.01,
        batch_size: int = 512,
        num_parts: int = 10,
        loader_type: str = 'cluster',
        update_graph_every: int = 10,
        warmup_epochs: int = 30,
        blend_alpha: float = 0.7,
        fusion_type: str = 'attention',
        reliability_alpha: float = 1.0,
        reliability_beta: float = 1.0,
        reliability_eta: float = 1.0,
        mask_seed: int = 42,
    ):
        """
        Args:
            dataset_name: Name of dataset ('Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'CS', 'Physics')
            dataset_root: Root directory for dataset storage
            mask_portion: Fraction of features to mask
            k: Number of neighbors for KNN similarity graph
            target_cluster_size: Target cluster size for graph partitioning
            hidden_dims: Hidden dimensions for GAT encoders
            latent_dim: Latent embedding dimension
            decoder_dims: Hidden dimensions for decoder
            lr: Learning rate
            tau: Temperature for contrastive loss
            beta_info: Weight for InfoNCE loss
            beta_barlow: Weight for Barlow Twins loss
            batch_size: Training batch size (only used if loader_type='neighbor')
            num_parts: Number of clusters/partitions (only used if loader_type='cluster')
            loader_type: Type of data loader ('cluster' or 'neighbor')
            update_graph_every: Update sim graph every N epochs (0 = no updates)
            warmup_epochs: Number of epochs before starting graph updates
            blend_alpha: Ratio of old edges to keep when updating (0.7 = 70% old, 30% new)
            fusion_type: Type of fusion ('attention' or 'reliability')
            reliability_alpha: Weight for missing rate in geo/sim views (only for reliability fusion)
            reliability_beta: Weight for missing rate in PPR view (only for reliability fusion)
            reliability_eta: Weight for neighbor consistency (only for reliability fusion)
            mask_seed: Random seed for masking
        """
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.mask_seed = mask_seed

        # Load dataset
        print(f"Loading dataset: {dataset_name}...")
        self.data = self.load_dataset(dataset_name, dataset_root)
        print(f"Dataset loaded: {self.data.num_nodes} nodes, {self.data.num_features} features")

        # Initialize data preparation pipeline
        self.pipeline = GraphPreparationPipeline(
            mask_portion=mask_portion,
            k=k,
            target_cluster_size=target_cluster_size,
            mask_seed=mask_seed,
        )

        # Initialize model
        self.model = MultiViewImputationModel(
            in_dim=self.data.num_features,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            decoder_dims=decoder_dims,
            lr=lr,
            tau=tau,
            beta_info=beta_info,
            beta_barlow=beta_barlow,
            batch_size=batch_size,
            num_parts=num_parts,
            loader_type=loader_type,
            update_graph_every=update_graph_every,
            warmup_epochs=warmup_epochs,
            blend_alpha=blend_alpha,
            fusion_type=fusion_type,
            reliability_alpha=reliability_alpha,
            reliability_beta=reliability_beta,
            reliability_eta=reliability_eta,
        )

    def load_dataset(self, name: str, root: str):
        """Load a benchmark graph dataset."""
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=f'{root}/{name}', name=name)
        elif name in ['Computers', 'Photo']:
            dataset = Amazon(root=f'{root}/Amazon{name}', name=name)
        elif name in ['CS', 'Physics']:
            dataset = Coauthor(root=f'{root}/Coauthor{name}', name=name)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        return dataset[0]

    def run(self, epochs: int = 200, eval_every: int = 10):
        """
        Run the complete pipeline: data preparation + training.

        Args:
            epochs: Number of training epochs
            eval_every: Evaluate every N epochs
        """
        print("\n" + "=" * 60)
        print(f"Multi-View Graph Imputation - {self.dataset_name}")
        print("=" * 60)

        # Step 1: Data preparation
        print("\nStep 1: Data Preparation")
        print("-" * 60)
        result = self.pipeline.run(self.data)

        print(f"Geo graph: {result['geo_graph'].num_nodes} nodes, {result['geo_graph'].edge_index.size(1)} edges")
        print(f"Sim graph: {result['sim_graph'].num_nodes} nodes, {result['sim_graph'].edge_index.size(1)} edges")
        print(f"Masked entries: {result['masked_positions'].sum().item()} / {result['masked_positions'].numel()}")
        print(f"Mask ratio: {result['masked_positions'].sum().item() / result['masked_positions'].numel():.3f}")

        # Step 2: Prepare model data
        print("\nStep 2: Model Preparation")
        print("-" * 60)
        self.model.prepare_data(result)

        # Step 3: Train model
        print("\nStep 3: Model Training")
        print("-" * 60)
        best_mae = self.model.train(epochs=epochs, eval_every=eval_every)

        return best_mae


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Training with ClusterLoader + Dynamic Graph Updates")
    print("=" * 60)

    trainer = ModelTrainer(
        dataset_name='Cora',
        mask_portion=0.3,
        k=20,
        target_cluster_size=500,
        hidden_dims=[32, 64],
        latent_dim=128,
        decoder_dims=[64, 32],
        lr=0.001,
        tau=0.5,
        beta_info=0.01,
        beta_barlow=0.01,
        num_parts=10,              # Number of clusters
        loader_type='cluster',     # Use ClusterLoader
        update_graph_every=10,     # Update sim graph every 10 epochs
        warmup_epochs=30,          # Start updates after 30 epochs
        blend_alpha=0.7,           # Keep 70% old edges, add 30% new
        mask_seed=42,
    )

    trainer.run(epochs=200, eval_every=10)

    #
    # trainer = ModelTrainer(
    #     dataset_name='Cora',
    #     mask_portion=0.3,
    #     batch_size=512,          # Batch size for NeighborLoader
    #     loader_type='neighbor',  # Use NeighborLoader
    # )
    # trainer.run(epochs=200, eval_every=10)

    # for dataset in ['Cora', 'CiteSeer']:
    #     print(f"\n{'='*60}")
    #     print(f"Running on {dataset}")
    #     print('='*60)
    #     trainer = ModelTrainer(
    #         dataset_name=dataset,
    #         mask_portion=0.3,
    #         loader_type='cluster',
    #         num_parts=10
    #     )
    #     trainer.run(epochs=100)
