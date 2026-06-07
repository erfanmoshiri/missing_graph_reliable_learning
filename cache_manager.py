"""
Cache manager for pre-computations: KNN imputation, reliability, PPR view, etc.
"""
import torch
import os
import hashlib
import json
from pathlib import Path


class PrecomputationCache:
    """
    Manages caching of expensive pre-computations.

    Cached items:
    - Masked features and masks
    - KNN-imputed features
    - Node reliability scores (r_self, r_nbr)
    - PPR edge index
    - Missingness-encoded features
    """

    def __init__(self, cache_dir='./cache'):
        """
        Args:
            cache_dir: directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _get_cache_key(self, dataset_name, miss_ratio, miss_mechanism, seed,
                      knn_k=5, ppr_k=10, ppr_alpha=0.15, overlap_penalty=0.1, variant=None):
        """
        Generate unique cache key for a configuration.

        Args:
            dataset_name: name of dataset
            miss_ratio: missingness ratio
            miss_mechanism: missingness mechanism (mcar, mar, mnar)
            seed: random seed
            knn_k: KNN k parameter
            ppr_k: PPR top-k parameter
            ppr_alpha: PPR alpha parameter
            overlap_penalty: overlap penalty for KNN
            variant: missingness variant (for MAR/MNAR)

        Returns:
            cache_key: unique string identifier
        """
        variant_str = f"_{variant}" if variant else ""
        config_str = f"{dataset_name}_{miss_ratio}_{miss_mechanism}{variant_str}_{seed}_" \
                    f"knn{knn_k}_ppr{ppr_k}_{ppr_alpha}_ovp{overlap_penalty}"

        # Use hash for shorter filename
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        return f"{dataset_name}_{cache_hash}"

    def _get_cache_path(self, cache_key):
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.pt"

    def _get_meta_path(self, cache_key):
        """Get path to metadata file."""
        return self.cache_dir / f"{cache_key}_meta.json"

    def exists(self, dataset_name, miss_ratio, miss_mechanism, seed, variant=None, **kwargs):
        """
        Check if cache exists for this configuration.

        Args:
            dataset_name, miss_ratio, miss_mechanism, seed: cache key params
            variant: missingness variant
            **kwargs: additional parameters (knn_k, ppr_k, etc.)

        Returns:
            bool: True if cache exists
        """
        cache_key = self._get_cache_key(dataset_name, miss_ratio, miss_mechanism,
                                       seed, variant=variant, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_meta_path(cache_key)

        return cache_path.exists() and meta_path.exists()

    def save(self, dataset_name, miss_ratio, miss_mechanism, seed,
             masked_features, mask, imputed_features,
             r_self, r_nbr, ppr_edge_index,
             knn_k=5, ppr_k=10, ppr_alpha=0.15, overlap_penalty=0.1, variant=None):
        """
        Save pre-computations to cache.

        Args:
            dataset_name: name of dataset
            miss_ratio: missingness ratio
            miss_mechanism: missingness mechanism
            seed: random seed
            masked_features: [n_nodes, n_features] masked feature matrix
            mask: [n_nodes, n_features] observation mask
            imputed_features: [n_nodes, n_features] imputed features
            r_self: [n_nodes] self reliability
            r_nbr: [n_nodes] neighborhood reliability
            ppr_edge_index: [2, n_edges] PPR graph edges
            knn_k, ppr_k, ppr_alpha, overlap_penalty: hyperparameters
            variant: missingness variant (for MAR/MNAR)
        """
        cache_key = self._get_cache_key(dataset_name, miss_ratio, miss_mechanism,
                                       seed, knn_k, ppr_k, ppr_alpha, overlap_penalty, variant)

        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_meta_path(cache_key)

        # Save tensors
        cache_data = {
            'masked_features': masked_features.cpu(),
            'mask': mask.cpu(),
            'imputed_features': imputed_features.cpu(),
            'r_self': r_self.cpu(),
            'r_nbr': r_nbr.cpu(),
            'ppr_edge_index': ppr_edge_index.cpu(),
        }

        torch.save(cache_data, cache_path)

        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'miss_ratio': miss_ratio,
            'miss_mechanism': miss_mechanism,
            'variant': variant,
            'seed': seed,
            'knn_k': knn_k,
            'ppr_k': ppr_k,
            'ppr_alpha': ppr_alpha,
            'overlap_penalty': overlap_penalty,
            'n_nodes': masked_features.shape[0],
            'n_features': masked_features.shape[1],
            'observed_ratio': mask.float().mean().item(),
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved precomputations to cache: {cache_key}")

    def load(self, dataset_name, miss_ratio, miss_mechanism, seed,
             knn_k=5, ppr_k=10, ppr_alpha=0.15, overlap_penalty=0.1,
             variant=None, device='cpu'):
        """
        Load pre-computations from cache.

        Args:
            dataset_name, miss_ratio, miss_mechanism, seed: cache key params
            knn_k, ppr_k, ppr_alpha, overlap_penalty: hyperparameters
            variant: missingness variant
            device: device to load tensors to

        Returns:
            dict with keys: masked_features, mask, imputed_features,
                          r_self, r_nbr, ppr_edge_index
            or None if cache doesn't exist
        """
        cache_key = self._get_cache_key(dataset_name, miss_ratio, miss_mechanism,
                                       seed, knn_k, ppr_k, ppr_alpha, overlap_penalty, variant)

        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_meta_path(cache_key)

        if not (cache_path.exists() and meta_path.exists()):
            return None

        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Load tensors
        cache_data = torch.load(cache_path, map_location=device)

        print(f"✓ Loaded precomputations from cache: {cache_key}")
        print(f"  Dataset: {metadata['dataset_name']}, "
              f"Nodes: {metadata['n_nodes']}, "
              f"Features: {metadata['n_features']}")
        print(f"  Observed ratio: {metadata['observed_ratio']:.3f}")

        return cache_data

    def list_cached(self):
        """
        List all cached configurations.

        Returns:
            list of metadata dicts
        """
        cached_configs = []

        for meta_path in self.cache_dir.glob("*_meta.json"):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            cached_configs.append(metadata)

        return cached_configs

    def clear(self, dataset_name=None):
        """
        Clear cache files.

        Args:
            dataset_name: if provided, only clear this dataset;
                         otherwise clear all
        """
        if dataset_name:
            pattern = f"{dataset_name}_*.pt"
            meta_pattern = f"{dataset_name}_*_meta.json"
        else:
            pattern = "*.pt"
            meta_pattern = "*_meta.json"

        count = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1

        for meta_file in self.cache_dir.glob(meta_pattern):
            meta_file.unlink()

        print(f"✓ Cleared {count} cache files")


def compute_and_cache(dataset, data, miss_ratio, miss_mechanism, seed,
                     knn_k=5, overlap_penalty=0.1,
                     ppr_k=10, ppr_alpha=0.15, ppr_iter=10,
                     variant=None, cache_dir='./cache', device='cpu'):
    """
    Compute pre-computations with caching.

    Checks cache first. If not found, computes and saves to cache.

    Args:
        dataset: PublicGraphDataset object
        data: graph data object
        miss_ratio: missingness ratio
        miss_mechanism: missingness mechanism
        seed: random seed
        knn_k, overlap_penalty: KNN parameters
        ppr_k, ppr_alpha, ppr_iter: PPR parameters
        variant: missingness variant (for MAR/MNAR)
        cache_dir: cache directory
        device: device for tensors

    Returns:
        dict with: masked_features, mask, imputed_features,
                  r_self, r_nbr, ppr_edge_index
    """
    from missingness import introduce_structured_missingness
    from initialization import mask_aware_knn_imputation, compute_node_reliability
    from ppr_view import ppr_topk_graph

    cache = PrecomputationCache(cache_dir=cache_dir)

    # Try to load from cache
    cached = cache.load(
        dataset.name, miss_ratio, miss_mechanism, seed,
        knn_k=knn_k, ppr_k=ppr_k, ppr_alpha=ppr_alpha,
        overlap_penalty=overlap_penalty, variant=variant, device=device
    )

    if cached is not None:
        return cached

    # Not in cache, compute from scratch
    print("Computing pre-computations (not in cache)...")

    # 1. Introduce missingness
    variant_str = f" ({variant})" if variant else ""
    print(f"  1/4 Introducing {miss_mechanism}{variant_str} missingness (target={miss_ratio})...")
    mask, masked_features = introduce_structured_missingness(
        data.x, data.edge_index, mechanism=miss_mechanism,
        target_ratio=miss_ratio, variant=variant, seed=seed
    )
    masked_features = masked_features.to(device)
    mask = mask.to(device)

    # 2. KNN imputation
    print(f"  2/4 Performing mask-aware KNN imputation (k={knn_k})...")
    imputed_features = mask_aware_knn_imputation(
        masked_features, mask, data.edge_index.to(device),
        k=knn_k, overlap_penalty=overlap_penalty
    )

    # 3. Compute reliability
    print(f"  3/4 Computing node reliability...")
    r_self, r_nbr, _ = compute_node_reliability(
        mask, data.edge_index.to(device), k=knn_k
    )

    # 4. Create PPR view
    print(f"  4/4 Creating PPR view (k={ppr_k}, alpha={ppr_alpha})...")
    ppr_edge_index = ppr_topk_graph(
        data.edge_index.to(device), data.num_nodes,
        k=ppr_k, alpha=ppr_alpha, max_iter=ppr_iter
    )

    # Save to cache
    cache.save(
        dataset.name, miss_ratio, miss_mechanism, seed,
        masked_features, mask, imputed_features,
        r_self, r_nbr, ppr_edge_index,
        knn_k=knn_k, ppr_k=ppr_k, ppr_alpha=ppr_alpha,
        overlap_penalty=overlap_penalty, variant=variant
    )

    return {
        'masked_features': masked_features,
        'mask': mask,
        'imputed_features': imputed_features,
        'r_self': r_self,
        'r_nbr': r_nbr,
        'ppr_edge_index': ppr_edge_index,
    }


if __name__ == '__main__':
    from datasets import PublicGraphDataset

    print("Testing cache manager...")

    # Load dataset
    dataset = PublicGraphDataset('cora')
    data = dataset.get_data()

    # Test compute and cache
    result1 = compute_and_cache(
        dataset, data, miss_ratio=0.3, miss_mechanism='mcar', seed=42,
        knn_k=5, ppr_k=10, ppr_alpha=0.15
    )

    print("\nFirst call completed. Now testing cache retrieval...")

    # Should load from cache
    result2 = compute_and_cache(
        dataset, data, miss_ratio=0.3, miss_mechanism='mcar', seed=42,
        knn_k=5, ppr_k=10, ppr_alpha=0.15
    )

    # Verify they match
    assert torch.equal(result1['mask'], result2['mask'])
    assert torch.equal(result1['imputed_features'], result2['imputed_features'])
    print("\n✓ Cache retrieval successful!")

    # List cached
    cache = PrecomputationCache()
    cached_list = cache.list_cached()
    print(f"\n✓ Found {len(cached_list)} cached configuration(s)")
    for cfg in cached_list:
        print(f"  - {cfg['dataset_name']}: {cfg['miss_mechanism']} "
              f"{cfg['miss_ratio']*100:.0f}% (seed={cfg['seed']})")
