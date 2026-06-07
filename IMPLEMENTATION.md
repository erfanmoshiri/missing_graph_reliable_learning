# Implementation Guide

This document provides a detailed overview of the implementation.

## Architecture Overview

```
Input: Graph with Missing Features
    ↓
[1] Mask-Aware KNN Imputation (initialization.py)
    → Computes distances only on commonly observed features
    → Imputes missing values from k-nearest neighbors
    ↓
[2] Missingness Pattern Encoding (model.py)
    → Concatenates: [features | mask | availability_pattern]
    → Projects to hidden dimension via MLP
    ↓
[3] Dual View Construction
    → Base View: Original graph adjacency
    → Global View: PPR-based diffusion graph (ppr_view.py)
    ↓
[4] Shared GAT Encoder (model.py)
    → Same encoder for both views
    → Multi-head attention with dropout
    ↓
[5] Output & Training
    → Base view → Classification head
    → Both views → Reliability-weighted Barlow Twins loss
    → Optional: Reconstruction loss
```

## File Descriptions

### `datasets.py`
- Loads public graph datasets (Cora, CiteSeer, PubMed, Amazon)
- Implements three missingness mechanisms:
  - **MCAR**: Missing Completely At Random
  - **MAR**: Missing At Random (depends on observed features)
  - **MNAR**: Missing Not At Random (depends on value itself)
- Ensures every node has at least one observed feature

### `initialization.py`
- **`mask_aware_knn_imputation()`**: KNN imputation with mask-aware distance
  - Only compares nodes on commonly observed dimensions
  - Adds overlap penalty to prefer neighbors with more shared features
  - Weighted average based on similarity
  
- **`compute_node_reliability()`**: Estimates node reliability
  - Self-reliability: fraction of observed features
  - Neighbor reliability: average observation rate in neighborhood
  - Learnable combination via sigmoid(rho)

- **`encode_missingness_pattern()`**: Creates missingness-aware input
  - Concatenates features, mask, and global availability pattern
  - Projects to encoder input dimension

### `ppr_view.py`
- **Personalized PageRank (PPR)** computation
  - Power iteration: P^(t+1) = (1-α)·A·P^(t) + α·I
  - Creates diffusion-based view of the graph
  
- **Top-k PPR graph construction**
  - Keeps only k highest PPR neighbors per node
  - Creates sparse alternative graph structure
  
- **`PPRViewConstructor`**: Module with caching
  - Computes PPR graph once and caches
  - Can force recomputation if needed

### `model.py`

#### `SharedGATEncoder`
- Multi-layer Graph Attention Network
- Shared between base and PPR views
- Architecture: GAT layers with ELU activation and dropout

#### `ReliabilityWeightedBarlowTwins`
- Implements the contrastive loss
- Normalizes embeddings to zero mean, unit variance
- Computes reliability-weighted cross-correlation matrix
- Loss = diagonal term + λ × off-diagonal term
  - Diagonal: encourages agreement between views
  - Off-diagonal: discourages redundant dimensions

#### `MissingGraphModel`
- End-to-end model combining all components
- Forward pass:
  1. Encode missingness pattern
  2. Encode both views with shared GAT
  3. Compute reliability from learned rho
  4. Classification from base view
- Supports optional reconstruction decoder

### `train.py`
- Training loop for single experiment
- Supports:
  - Classification task with cross-entropy
  - Reliability-weighted Barlow Twins contrastive loss
  - Optional reconstruction loss
- Early stopping based on validation accuracy
- Model checkpointing

### `run_experiments.py`
- Runs multiple experiments programmatically
- Three modes:
  - **quick**: Single dataset, one setting, fast test
  - **full**: All datasets, all missingness settings
  - **ablation**: Tests key components (with/without contrastive, different beta, etc.)
- Computes mean ± std over multiple seeds
- Saves results to JSON

### `test_pipeline.py`
- Comprehensive test of all components
- Runs full forward and backward pass
- Validates shapes and loss computation
- Quick sanity check before running experiments

## Key Implementation Details

### 1. Mask-Aware Distance

```python
# Overlap mask: which features both nodes have
o_ij = mask_i * mask_j
c_ij = o_ij.sum()  # number of commonly observed features

# Distance only on shared features
diff = (features_i - features_j) * o_ij
d_ij = (diff ** 2).sum() / (c_ij + eps)

# Add penalty for small overlap
d_ij_prime = d_ij + lambda_o / (c_ij + eps)
```

### 2. Missingness Pattern Encoding

```python
# Global frequency of each feature
q = mask.mean(dim=0)  # [n_features]

# Node-specific availability pattern
s_i = mask_i * q  # element-wise, [n_features]

# Input to encoder
h0 = MLP([features_i, mask_i, s_i])  # concatenated
```

### 3. Reliability Computation

```python
# Self-observation reliability
r_self = mask.mean(dim=1)  # fraction observed per node

# Neighbor reliability (computed during KNN)
r_nbr = ...  # average observation in neighborhood

# Learnable combination
weight = sigmoid(rho)  # rho is a learnable parameter
reliability = weight * r_self + (1 - weight) * r_nbr
```

### 4. Reliability-Weighted Barlow Twins

```python
# Normalize embeddings
z_base_norm = (z_base - z_base.mean(0)) / z_base.std(0)
z_ppr_norm = (z_ppr - z_ppr.mean(0)) / z_ppr.std(0)

# Weighted cross-correlation
C = (z_base_norm.T @ (reliability * z_ppr_norm)) / reliability.sum()

# Loss
diagonal_loss = ((1 - diag(C)) ** 2).sum()
off_diagonal_loss = (C[off_diag_mask] ** 2).sum()
loss = diagonal_loss + lambda_bt * off_diagonal_loss
```

## Hyperparameters

### Critical Hyperparameters

1. **`beta`** (contrastive loss weight): 0.1 - 5.0
   - Higher: more emphasis on view consistency
   - Lower: more emphasis on task performance
   - Default: 1.0

2. **`lambda_bt`** (Barlow Twins off-diagonal): 0.001 - 0.01
   - Controls redundancy reduction
   - Default: 0.005

3. **`ppr_k`** (PPR neighbors): 5 - 20
   - Number of diffusion neighbors per node
   - Default: 10

4. **`ppr_alpha`** (PPR teleport): 0.1 - 0.2
   - Lower: more diffusion, longer range
   - Higher: more local, closer to base graph
   - Default: 0.15

### Model Architecture

- **`hidden_dim`**: 64 - 256 (default: 128)
- **`embed_dim`**: 32 - 128 (default: 64)
- **`num_layers`**: 2 - 3 (default: 2)
- **`heads`**: 4 - 8 (default: 8)
- **`dropout`**: 0.5 - 0.7 (default: 0.6)

### Training

- **`lr`**: 0.001 - 0.01 (default: 0.005)
- **`weight_decay`**: 1e-4 - 1e-3 (default: 5e-4)
- **`epochs`**: 200 - 500 (default: 500)
- **`patience`**: 20 - 50 (default: 50)

## Running Experiments

### Quick Test
```bash
python test_pipeline.py --test pipeline
```

### Single Experiment
```bash
python train.py \
  --dataset cora \
  --miss_ratio 0.3 \
  --miss_mechanism mcar \
  --epochs 500 \
  --save_model
```

### Systematic Benchmark
```bash
# Quick test (3 seeds)
python run_experiments.py --mode quick --datasets cora --num_seeds 3

# Full benchmark (5 seeds, all datasets)
python run_experiments.py --mode full --num_seeds 5

# Ablation studies
python run_experiments.py --mode ablation --datasets cora --num_seeds 3
```

### Hyperparameter Tuning

```bash
# Try different beta values
for beta in 0.1 0.5 1.0 2.0 5.0; do
  python train.py --dataset cora --miss_ratio 0.3 --beta $beta --save_model
done

# Try different PPR settings
for k in 5 10 15 20; do
  for alpha in 0.1 0.15 0.2; do
    python train.py --dataset cora --ppr_k $k --ppr_alpha $alpha
  done
done
```

## Troubleshooting

### Out of Memory
- Reduce `hidden_dim`, `embed_dim`, or `heads`
- Use smaller batch (implicit in full-graph training)
- For large graphs, consider mini-batching (requires code modification)

### Poor Performance
- Check if missingness ratio is too high (>0.7 can be very challenging)
- Tune `beta` (try 0.1, 0.5, 1.0, 2.0, 5.0)
- Increase `ppr_k` for better global view
- Try different `dropout` values (0.5 - 0.7)

### Slow Training
- Reduce `ppr_iter` (10 is usually sufficient)
- Use fewer `heads` (4 instead of 8)
- Reduce `num_layers` (2 is often enough)

### Unstable Training
- Lower learning rate
- Increase `weight_decay`
- Use gradient clipping (add to train.py)

## Extension Ideas

1. **Mini-batch training** for very large graphs
2. **Additional views**: k-hop subgraph, feature similarity graph
3. **Attention-based view fusion** instead of separate encoders
4. **Self-training** for unlabeled nodes
5. **Adaptive reliability** that updates during training
6. **Multi-task learning** with auxiliary tasks

## Performance Notes

- Cora (2.7K nodes): ~1-2 min/epoch on CPU, ~5-10 sec/epoch on GPU
- PubMed (19K nodes): ~10-20 min/epoch on CPU, ~30-60 sec/epoch on GPU
- Amazon (13K-14K nodes): ~5-15 min/epoch on CPU, ~20-40 sec/epoch on GPU

GPU highly recommended for larger datasets.
