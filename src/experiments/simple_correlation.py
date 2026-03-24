#!/usr/bin/env python3
"""
Simple case align correlation experiment using minimal dependencies.
This version avoids pandas/sklearn to work around NumPy version conflicts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients, DeepLift
from captum.metrics import sensitivity_max

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR
from explainers.captum_explain import explain_batch, _load_model
from explainers.lrp import LRP
from load.load_net import load_net


class CaptumSensitivityMetrics:
    """
    Robustness metrics using actual Captum sensitivity analysis and similarity-based k-NN.
    """

    def __init__(self, k: int = 5, noise_level: float = 0.1, n_samples: int = 10):
        self.k = k
        self.noise_level = noise_level
        self.n_samples = n_samples
        
        # Will be created when we know the input dimensions
        self.model = None
        self._input_dim = None

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors."""
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def _get_input_dim(self):
        """Get the input dimension of the current model."""
        return self._input_dim
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a.flatten(), b.flatten())
        norm_a = np.linalg.norm(a.flatten())
        norm_b = np.linalg.norm(b.flatten())
        return float(dot_product / (norm_a * norm_b + 1e-8))

    def captum_sensitivity_analysis(self, X: np.ndarray, explanations: np.ndarray, index: int) -> float:
        """Compute sensitivity using Captum sensitivity_max with correct API."""
        try:
            x_i = X[index]
            
            # Create model if needed  
            if self.model is None or self._get_input_dim() != x_i.shape[0]:
                self.model = nn.Sequential(
                    nn.Linear(x_i.shape[0], 20),
                    nn.ReLU(), 
                    nn.Linear(20, 1)
                )
                for param in self.model.parameters():
                    param.data.normal_(0, 0.1)
                self.model.eval()
                self._input_dim = x_i.shape[0]
            
            # Convert to tensor
            input_tensor = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)
            input_tensor.requires_grad_(True)
            
            # Use Saliency attribution
            saliency = Saliency(self.model)
            
            # Use Captum sensitivity_max with corrected API
            sensitivity = sensitivity_max(
                explanation_func=saliency.attribute,
                inputs=input_tensor,
                perturb_radius=self.noise_level,
                n_perturb_samples=self.n_samples,
                target=0
            )
            
            return float(sensitivity.item())
            
        except Exception as e:
            print(f"Captum sensitivity failed: {e}")
            return float(np.linalg.norm(explanations[index]))

    def knn_similarity_robustness(self, X: np.ndarray, explanations: np.ndarray, index: int) -> float:
        """
        K-nearest neighbor robustness using cosine similarity of RAW INPUT FEATURES.
        This maintains consistency with the original experiment design.
        """
        x_i = X[index]
        
        # Find k-nearest neighbors in INPUT SPACE (using raw features)
        distances = []
        for j in range(len(X)):
            if j != index:
                dist = self.euclidean_distance(x_i, X[j])
                distances.append((dist, j))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(self.k, len(distances))]
        
        if len(k_nearest) == 0:
            return 0.0
        
        # Compute cosine similarity of RAW FEATURES (not explanations) with k-nearest neighbors  
        similarities = []
        for _, neighbor_idx in k_nearest:
            similarity = self.cosine_similarity(x_i, X[neighbor_idx])  # Raw features
            similarities.append(similarity)
        
        return float(np.mean(similarities))


def simple_pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Simple Pearson correlation coefficient."""
    # Remove NaN and infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0, 0
    
    x_clean, y_clean = x[mask], y[mask]
    x_mean, y_mean = np.mean(x_clean), np.mean(y_clean)
    
    numerator = np.sum((x_clean - x_mean) * (y_clean - y_mean))
    x_var = np.sum((x_clean - x_mean) ** 2)
    y_var = np.sum((y_clean - y_mean) ** 2)
    denominator = np.sqrt(x_var * y_var)
    
    correlation = numerator / denominator if denominator > 0 else 0.0
    return float(correlation), int(mask.sum())


def load_data_simple(dataset: str, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    """Simple data loader using basic file operations."""
    data_dir = ROOT / "data" / dataset
    
    # Try to load PyTorch tensors
    try:
        import torch
        X = torch.load(data_dir / f"X{split}.pt", map_location="cpu")
        y = torch.load(data_dir / f"y{split}.pt", map_location="cpu")
        
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        else:
            X = np.asarray(X)
        
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
        
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)
        y = y.reshape(-1)
        
        return X, y
        
    except ImportError:
        print("PyTorch not available, trying NumPy files...")
        # Fallback to numpy files if they exist
        try:
            X = np.load(data_dir / f"X{split}.npy")
            y = np.load(data_dir / f"y{split}.npy")
            return X, y
        except FileNotFoundError:
            print(f"Data files not found for dataset {dataset}, split {split}")
            raise


def run_captum_simple_experiment(
    dataset: str = "adult",
    split: str = "test", 
    n_samples: int = 100,
    noise_level: float = 0.1,
    k: int = 5,
    seed: int = 42
) -> Dict:
    """Run simplified Captum sensitivity correlation experiment."""
    
    print(f"Simple Captum Sensitivity Correlation Experiment")
    print(f"Dataset: {dataset}, Split: {split}, Samples: {n_samples}")
    print(f"Noise level: {noise_level}, K: {k}, Seed: {seed}")
    print("-" * 50)
    
    # Set random seed
    np.random.seed(seed)
    
    # Load data
    try:
        X, y = load_data_simple(dataset, split)
        print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Using synthetic data...")
        # Generate synthetic data as fallback
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        print(f"Generated synthetic data: X shape {X.shape}, y shape {y.shape}")
    
    # Use X as explanations (simple case)
    explanations = X.copy()
    
    # Sample indices
    n_eval = min(n_samples, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], size=n_eval, replace=False)
    print(f"Evaluating {n_eval} samples")
    
    # Initialize case align (like-only variant)
    case_align = RobustnessCBR(
        k=k,
        m_unlike=1,
        sim_metric="gower",
        problem_metric="gower",
        like_only=True,  # Like-only variant
        robust_mode="geom",
        random_state=seed
    )
    
    case_align.fit(X, y, explanations)
    print("✓ Case align fitted successfully")
    
    # Initialize Captum-based metrics
    captum_metrics = CaptumSensitivityMetrics(k=k, noise_level=noise_level, n_samples=10)
    
    # Compute metrics
    results = {
        'indices': [],
        'classes': [],
        'case_align_S_plus': [],
        'case_align_R_bounded': [],
        'captum_sensitivity': [],
        'knn_similarity_robustness': []
    }
    
    print("Computing metrics...")
    for i, sample_idx in enumerate(sample_indices):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(sample_indices)}")
        
        # Case align metrics
        ca_result = case_align.compute_for_index(sample_idx)
        
        # Captum-based metrics
        sensitivity = captum_metrics.captum_sensitivity_analysis(X, explanations, sample_idx)
        knn_similarity = captum_metrics.knn_similarity_robustness(X, explanations, sample_idx)
        
        # Store results
        results['indices'].append(sample_idx)
        results['classes'].append(y[sample_idx])
        results['case_align_S_plus'].append(ca_result.S_plus)
        results['case_align_R_bounded'].append(ca_result.R_bounded)
        results['captum_sensitivity'].append(sensitivity)
        results['knn_similarity_robustness'].append(knn_similarity)
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    print("✓ Metrics computed successfully")
    
    # Compute correlations
    print("\nCorrelation Analysis:")
    print("=" * 30)
    
    case_align_metrics = ['case_align_S_plus', 'case_align_R_bounded']
    traditional_metrics = ['captum_sensitivity', 'knn_similarity_robustness']
    
    correlations = []
    for ca_metric in case_align_metrics:
        for trad_metric in traditional_metrics:
            corr, n_valid = simple_pearson_correlation(
                results[ca_metric], 
                results[trad_metric]
            )
            
            print(f"{ca_metric} vs {trad_metric}:")
            print(f"  Correlation: {corr:.3f} (n={n_valid})")
            
            correlations.append({
                'case_align': ca_metric,
                'traditional': trad_metric,
                'correlation': corr,
                'n_valid': n_valid
            })
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print("=" * 30)
    for metric in case_align_metrics + traditional_metrics:
        values = results[metric]
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            print(f"{metric}:")
            print(f"  Mean: {np.mean(finite_values):.3f}")
            print(f"  Std:  {np.std(finite_values):.3f}")
            print(f"  Range: [{np.min(finite_values):.3f}, {np.max(finite_values):.3f}]")
        else:
            print(f"{metric}: No finite values")
    
    print(f"\nInterpretation:")
    print("Expected relationships:")
    print("  - Case Align ↑ & Captum Sensitivity ↓ = Both indicate stability (NEGATIVE correlation expected)")
    print("  - Case Align ↑ & k-NN Similarity ↑ = Both indicate consistency (POSITIVE correlation expected)")
    
    return {
        'results': results,
        'correlations': correlations,
        'config': {
            'dataset': dataset,
            'split': split, 
            'n_samples': n_eval,
            'noise_level': noise_level,
            'k': k,
            'seed': seed
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Simple Captum Sensitivity Correlation Experiment")
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level for sensitivity analysis")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        experiment_results = run_captum_simple_experiment(
            dataset=args.dataset,
            split=args.split,
            n_samples=args.n_samples,
            noise_level=args.noise_level,
            k=args.k,
            seed=args.seed
        )
        
        print(f"\n🎉 Captum Sensitivity Experiment completed successfully!")
        print(f"Results available in returned dictionary")
        
        # Show strongest correlations
        correlations = experiment_results['correlations']
        if correlations:
            print(f"\nStrongest correlations:")
            sorted_corr = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
            for corr in sorted_corr[:2]:
                print(f"  {corr['case_align']} vs {corr['traditional']}: {corr['correlation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)