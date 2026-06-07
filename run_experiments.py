"""
Run experiments across multiple datasets and missingness settings.
"""
import subprocess
import numpy as np
import json
import os
from datetime import datetime


def run_experiment(dataset, miss_ratio, miss_mechanism, seed, args_dict=None):
    """
    Run a single experiment.

    Args:
        dataset: dataset name
        miss_ratio: missingness ratio
        miss_mechanism: missingness mechanism
        seed: random seed
        args_dict: additional arguments as dict

    Returns:
        test_accuracy: float
    """
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--miss_ratio', str(miss_ratio),
        '--miss_mechanism', miss_mechanism,
        '--seed', str(seed)
    ]

    # Add additional arguments
    if args_dict:
        for key, value in args_dict.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse test accuracy from output
        for line in output.split('\n'):
            if 'Test accuracy:' in line:
                acc = float(line.split(':')[1].strip())
                return acc

        return None

    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def run_full_benchmark(datasets=None, miss_ratios=None, miss_mechanisms=None,
                      num_seeds=5, args_dict=None):
    """
    Run full benchmark experiments.

    Args:
        datasets: list of dataset names
        miss_ratios: list of missingness ratios
        miss_mechanisms: list of missingness mechanisms
        num_seeds: number of random seeds
        args_dict: additional training arguments

    Returns:
        results: nested dict of results
    """
    if datasets is None:
        datasets = ['cora', 'citeseer', 'pubmed', 'computers', 'photo']

    if miss_ratios is None:
        miss_ratios = [0.1, 0.3, 0.5, 0.7]

    if miss_mechanisms is None:
        miss_mechanisms = ['mcar', 'mar', 'mnar']

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset in datasets:
        results[dataset] = {}

        for mechanism in miss_mechanisms:
            results[dataset][mechanism] = {}

            for ratio in miss_ratios:
                print(f"\n{'='*80}")
                print(f"Dataset: {dataset}, Mechanism: {mechanism}, Ratio: {ratio}")
                print('='*80)

                accs = []
                for seed in range(num_seeds):
                    print(f"\nSeed {seed + 1}/{num_seeds}")
                    acc = run_experiment(dataset, ratio, mechanism, seed, args_dict)

                    if acc is not None:
                        accs.append(acc)
                        print(f"Test accuracy: {acc:.4f}")

                if accs:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    results[dataset][mechanism][ratio] = {
                        'mean': mean_acc,
                        'std': std_acc,
                        'all': accs
                    }
                    print(f"\nResults: {mean_acc:.4f} ± {std_acc:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = f'results/benchmark_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results


def print_results_table(results):
    """
    Print results in a formatted table.

    Args:
        results: results dict from run_full_benchmark
    """
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)

    for dataset in results:
        print(f"\n{dataset.upper()}")
        print("-" * 100)
        print(f"{'Mechanism':<12} {'Ratio':<8} {'Accuracy':<20}")
        print("-" * 100)

        for mechanism in results[dataset]:
            for ratio in sorted(results[dataset][mechanism].keys()):
                res = results[dataset][mechanism][ratio]
                mean = res['mean']
                std = res['std']
                print(f"{mechanism:<12} {ratio:<8.1f} {mean:.4f} ± {std:.4f}")

    print("="*120)


def run_ablation_studies(dataset='cora', miss_ratio=0.3, mechanism='mcar', num_seeds=3):
    """
    Run ablation studies on key components.

    Tests:
    1. Without contrastive loss (beta=0)
    2. Without reliability weighting (set rho to fixed value)
    3. Without PPR view (only base view)
    4. With reconstruction loss (gamma>0)
    """
    print("\n" + "="*80)
    print("ABLATION STUDIES")
    print("="*80)

    ablations = {
        'Full model': {'beta': 1.0, 'gamma': 0.0},
        'No contrastive': {'beta': 0.0, 'gamma': 0.0},
        'With reconstruction': {'beta': 1.0, 'gamma': 0.5},
        'Lower beta': {'beta': 0.1, 'gamma': 0.0},
        'Higher beta': {'beta': 5.0, 'gamma': 0.0},
    }

    results = {}

    for name, args_dict in ablations.items():
        print(f"\n{name}:")
        print(f"  Args: {args_dict}")

        accs = []
        for seed in range(num_seeds):
            acc = run_experiment(dataset, miss_ratio, mechanism, seed, args_dict)
            if acc is not None:
                accs.append(acc)

        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[name] = {'mean': mean_acc, 'std': std_acc, 'all': accs}
            print(f"  Result: {mean_acc:.4f} ± {std_acc:.4f}")

    # Save ablation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    with open(f'results/ablation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark experiments')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'ablation'],
                       help='Experiment mode')
    parser.add_argument('--datasets', nargs='+', default=['cora'],
                       help='Datasets to test')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of random seeds')

    args = parser.parse_args()

    if args.mode == 'quick':
        # Quick test on one dataset, one setting
        print("Running quick test...")
        results = run_full_benchmark(
            datasets=args.datasets,
            miss_ratios=[0.3],
            miss_mechanisms=['mcar'],
            num_seeds=args.num_seeds,
            args_dict={'epochs': 200, 'patience': 20}
        )
        print_results_table(results)

    elif args.mode == 'full':
        # Full benchmark
        print("Running full benchmark...")
        results = run_full_benchmark(
            datasets=['cora', 'citeseer', 'pubmed', 'computers', 'photo'],
            miss_ratios=[0.1, 0.3, 0.5, 0.7],
            miss_mechanisms=['mcar', 'mar', 'mnar'],
            num_seeds=5,
            args_dict={'save_model': True}
        )
        print_results_table(results)

    elif args.mode == 'ablation':
        # Ablation studies
        print("Running ablation studies...")
        results = run_ablation_studies(
            dataset=args.datasets[0],
            miss_ratio=0.3,
            mechanism='mcar',
            num_seeds=args.num_seeds
        )
        print("\nAblation Results:")
        for name, res in results.items():
            print(f"{name}: {res['mean']:.4f} ± {res['std']:.4f}")
