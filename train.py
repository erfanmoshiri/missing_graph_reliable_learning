import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import os

from datasets import PublicGraphDataset
from cache_manager import compute_and_cache
from model import MissingGraphModel


def train_epoch(model, optimizer, data, imputed_features, mask, ppr_edge_index,
                r_self, r_nbr, train_mask, beta=1.0, gamma=0.0):
    """
    Train for one epoch.

    Args:
        model: the model
        optimizer: optimizer
        data: graph data object
        imputed_features: imputed feature matrix
        mask: observation mask
        ppr_edge_index: PPR graph edges
        r_self: self reliability
        r_nbr: neighborhood reliability
        train_mask: training mask
        beta: weight for contrastive loss
        gamma: weight for reconstruction loss (if enabled)

    Returns:
        loss, task_loss, contrastive_loss
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z_base, z_ppr, logits, reliability = model(
        imputed_features, mask,
        data.edge_index, ppr_edge_index,
        r_self, r_nbr
    )

    # Task loss (classification)
    task_loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

    # Contrastive loss (on all nodes or just training nodes)
    contrastive_loss = model.model.contrastive_loss(
        z_base[train_mask], z_ppr[train_mask], reliability[train_mask]
    )

    # Optional reconstruction loss
    if gamma > 0 and model.model.use_reconstruction:
        recon_loss = model.model.reconstruction_loss(z_base, data.x, mask)
        total_loss = task_loss + beta * contrastive_loss + gamma * recon_loss
    else:
        recon_loss = torch.tensor(0.0)
        total_loss = task_loss + beta * contrastive_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), task_loss.item(), contrastive_loss.item()


@torch.no_grad()
def evaluate(model, data, imputed_features, mask, ppr_edge_index,
             r_self, r_nbr, split_mask):
    """
    Evaluate model on a split.

    Args:
        model: the model
        data: graph data
        imputed_features: imputed features
        mask: observation mask
        ppr_edge_index: PPR graph
        r_self, r_nbr: reliability components
        split_mask: mask for evaluation split

    Returns:
        accuracy, loss
    """
    model.eval()

    z_base, z_ppr, logits, reliability = model(
        imputed_features, mask,
        data.edge_index, ppr_edge_index,
        r_self, r_nbr
    )

    # Loss
    loss = F.cross_entropy(logits[split_mask], data.y[split_mask])

    # Accuracy
    pred = logits[split_mask].argmax(dim=1)
    acc = (pred == data.y[split_mask]).float().mean()

    return acc.item(), loss.item()


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    dataset = PublicGraphDataset(args.dataset, normalize=True)
    data = dataset.get_data()
    train_mask, val_mask, test_mask = dataset.get_split_masks()

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {dataset.num_features}, Classes: {dataset.num_classes}")
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Compute or load pre-computations (with caching)
    variant_str = f" ({args.miss_variant})" if args.miss_variant else ""
    print(f"\nPre-computations (miss_ratio={args.miss_ratio}, mechanism={args.miss_mechanism}{variant_str})...")
    if args.no_cache:
        # Force recompute
        from missingness import introduce_structured_missingness
        from initialization import mask_aware_knn_imputation, compute_node_reliability
        from ppr_view import ppr_topk_graph

        print("  Caching disabled, computing from scratch...")
        mask, masked_features = introduce_structured_missingness(
            data.x, data.edge_index, mechanism=args.miss_mechanism,
            target_ratio=args.miss_ratio, variant=args.miss_variant, seed=args.seed
        )
        imputed_features = mask_aware_knn_imputation(
            masked_features, mask, data.edge_index,
            k=args.knn_k, overlap_penalty=args.overlap_penalty
        )
        r_self, r_nbr, _ = compute_node_reliability(mask, data.edge_index, k=args.knn_k)
        ppr_edge_index = ppr_topk_graph(
            data.edge_index, data.num_nodes,
            k=args.ppr_k, alpha=args.ppr_alpha, max_iter=args.ppr_iter
        )
    else:
        # Use caching
        precomp = compute_and_cache(
            dataset, data,
            miss_ratio=args.miss_ratio,
            miss_mechanism=args.miss_mechanism,
            seed=args.seed,
            knn_k=args.knn_k,
            overlap_penalty=args.overlap_penalty,
            ppr_k=args.ppr_k,
            ppr_alpha=args.ppr_alpha,
            ppr_iter=args.ppr_iter,
            variant=args.miss_variant,
            cache_dir=args.cache_dir,
            device='cpu'  # Load to CPU first, then move to device
        )
        masked_features = precomp['masked_features']
        mask = precomp['mask']
        imputed_features = precomp['imputed_features']
        r_self = precomp['r_self']
        r_nbr = precomp['r_nbr']
        ppr_edge_index = precomp['ppr_edge_index']

    # Move to device
    masked_features = masked_features.to(device)
    mask = mask.to(device)
    imputed_features = imputed_features.to(device)
    r_self = r_self.to(device)
    r_nbr = r_nbr.to(device)
    ppr_edge_index = ppr_edge_index.to(device)
    data = data.to(device)
    print(f"Observed ratio: {mask.float().mean():.3f}")
    print(f"PPR graph edges: {ppr_edge_index.shape[1]}")

    # Initialize model
    print("\nInitializing model...")
    model = MissingGraphModel(
        n_features=dataset.num_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        lambda_bt=args.lambda_bt
    ).to(device)

    # Enable reconstruction if requested
    if args.gamma > 0:
        model.model.enable_reconstruction(dataset.num_features)
        print("Reconstruction enabled")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        loss, task_loss, contrastive_loss = train_epoch(
            model, optimizer, data,
            imputed_features, mask, ppr_edge_index,
            r_self, r_nbr, train_mask,
            beta=args.beta, gamma=args.gamma
        )

        # Evaluate
        if epoch % args.eval_every == 0:
            train_acc, _ = evaluate(model, data, imputed_features, mask,
                                   ppr_edge_index, r_self, r_nbr, train_mask)
            val_acc, val_loss = evaluate(model, data, imputed_features, mask,
                                        ppr_edge_index, r_self, r_nbr, val_mask)

            print(f"Epoch {epoch:03d}: Loss={loss:.4f} (task={task_loss:.4f}, bt={contrastive_loss:.4f}) | "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                if args.save_model:
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(model.state_dict(),
                             f'checkpoints/{args.dataset}_miss{args.miss_ratio}_{args.miss_mechanism}.pt')
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    if args.save_model:
        model.load_state_dict(
            torch.load(f'checkpoints/{args.dataset}_miss{args.miss_ratio}_{args.miss_mechanism}.pt')
        )

    test_acc, test_loss = evaluate(model, data, imputed_features, mask,
                                  ppr_edge_index, r_self, r_nbr, test_mask)
    print(f"Test accuracy: {test_acc:.4f}")

    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contrastive Graph Learning with Missing Features')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cora',
                       choices=['cora', 'citeseer', 'pubmed', 'computers', 'photo'],
                       help='Dataset name')

    # Missingness
    parser.add_argument('--miss_ratio', type=float, default=0.5,
                       help='Target fraction of missing values (default 0.5 = 50%)')
    parser.add_argument('--miss_mechanism', type=str, default='mcar',
                       choices=['mcar', 'mar', 'mnar'],
                       help='Missingness mechanism')
    parser.add_argument('--miss_variant', type=str, default=None,
                       help='Missingness variant: MAR (degree|observation), MNAR (frequency|rare|value)')

    # Initialization
    parser.add_argument('--knn_k', type=int, default=5,
                       help='Number of neighbors for KNN imputation')
    parser.add_argument('--overlap_penalty', type=float, default=0.1,
                       help='Overlap penalty for mask-aware distance')

    # PPR view
    parser.add_argument('--ppr_k', type=int, default=10,
                       help='Top-k PPR neighbors')
    parser.add_argument('--ppr_alpha', type=float, default=0.15,
                       help='PPR teleport probability')
    parser.add_argument('--ppr_iter', type=int, default=10,
                       help='PPR iteration count')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GAT layers')
    parser.add_argument('--heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6,
                       help='Dropout probability')

    # Loss weights
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for contrastive loss')
    parser.add_argument('--gamma', type=float, default=0.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda_bt', type=float, default=0.005,
                       help='Barlow Twins off-diagonal penalty')

    # Training
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--eval_every', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true',
                       help='Save best model checkpoint')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Caching
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable caching (always recompute)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory')
    parser.add_argument('--clear_cache', type=str, default=None,
                       help='Clear cache for dataset (or "all")')

    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        from cache_manager import PrecomputationCache
        cache = PrecomputationCache(cache_dir=args.cache_dir)
        if args.clear_cache == 'all':
            cache.clear()
        else:
            cache.clear(dataset_name=args.clear_cache)
        exit(0)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    test_acc = main(args)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
