# Contrastive Graph Learning with Missing Features

A graph neural network that learns robust node representations when features are missing.

## What It Does

Standard GNNs fail when node features are missing. This model:
- Handles **structured missingness** (50% missing features)
- Uses **two graph views**: base graph + PPR diffusion graph
- Weights learning by **node reliability** (how much data each node has)
- Learns with **contrastive loss** (Barlow Twins) instead of requiring labels

## Quick Start

```bash
# Install dependencies (with your venv activated)
pip install -r requirements.txt

# Train on Cora dataset with 50% missing features
python train.py --dataset cora --miss_ratio 0.5 --miss_mechanism mcar --epochs 500

# Try different missingness patterns
python train.py --dataset cora --miss_ratio 0.5 --miss_mechanism mar --miss_variant degree
python train.py --dataset cora --miss_ratio 0.5 --miss_mechanism mnar --miss_variant frequency
```

## Datasets

Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo

## Key Parameters

```bash
--miss_ratio 0.5              # 50% missing (default)
--miss_mechanism mcar         # mcar, mar, mnar
--miss_variant degree         # for MAR: degree, observation
                              # for MNAR: frequency, rare, value
--beta 1.0                    # contrastive loss weight
--hidden_dim 128              # model size
--epochs 500                  # training iterations
```

## How It Works

1. **Impute** missing values with mask-aware KNN (only uses shared observed features)
2. **Encode** missingness pattern (which features are missing matters)
3. **Two views**: original graph + PPR global view
4. **Learn** by aligning the two views with reliability-weighted Barlow Twins
5. **Classify** from learned representations

## Citation

```bibtex
@article{yourpaper2024,
  title={Contrastive Graph Learning under Missing Node Attributes with Reliability-Aware Fusion},
  author={Your Name},
  year={2024}
}
```
