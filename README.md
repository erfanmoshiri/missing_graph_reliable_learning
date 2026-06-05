# Contrastive Graph Learning Under Structured Missingness: A Reliability-Aware Approach
Attributes with Reliability-Aware Fusion



This repository contains the implementation for **Contrastive Multi-Relational Graph Learning** — a model proposed in the paper:

> **Contrastive Graph Learning under Missing Node Attributes with Reliability-Aware Fusion**  

# Contrastive Graph Learning under Missing Attributes with Reliability-Aware Fusion

## Overview
This repository contains the code for our paper on learning graph representations when node attributes are missing. Instead of relying on imputation first, the method learns robust node embeddings directly from incomplete data and uses reliability-aware fusion to combine multiple graph views. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Problem
Missing node attributes are common in real-world graphs and can hurt both representation learning and downstream tasks. This is especially important in property data, where missingness is often structured rather than fully random. Existing graph contrastive methods often use stochastic augmentations or simple fusion, which can be weak under attribute missingness. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## Key Components
- Base graph plus a auxiliary relation graphs
- Relation-specific graph encoders
- Reliability-aware node-wise fusion
- Self-supervised training with Barlow Twins

## Datasets
- Cora
- CiteSeer
- PubMed
- Amazon Computers
- Amazon Photo
- Australian property dataset for downstream property valuation.

## Summary of Results
The method consistently improves representation quality and downstream performance under MCAR, MAR, and MNAR missingness settings. On the property valuation task, it outperforms strong baselines across benchmark settings and shows stable gains in ablation studies.
