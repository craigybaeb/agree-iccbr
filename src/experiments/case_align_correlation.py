#!/usr/bin/env python3
"""
Experiment to analyze correlation between case align (like-only variant) 
and Captum-based robustness metrics (sensitivity analysis and k-NN similarity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR

# Import Captum for sensitivity analysis
try:
    import torch
    from captum.attr import NoiseTunnel, IntegratedGradients, Saliency
    from captum.metrics import sensitivity_max
    CAPTUM_AVAILABLE = True
    print("✓ Captum available for sensitivity analysis")
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠ Captum not available, will use alternative sensitivity metric")


class CaptumSensitivityMetrics:
    """
    Robustness metrics using actual Captum sensitivity analysis and similarity-based k-NN.
    """
    
    def __init__(self, k: int = 5, noise_level: float = 0.1, n_samples: int = 10, same_class_only: bool = False, similarity_metric: str = "euclidean"):
        self.k = k
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.same_class_only = same_class_only
        self.similarity_metric = similarity_metric
        
        # Simple linear model for Captum analysis (will be resized based on input)
        if CAPTUM_AVAILABLE:
            import torch.nn as nn
            self.model = nn.Linear(1, 1)
            self.model.eval()
        else:
            self.model = None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a.flatten(), b.flatten())
        norm_a = np.linalg.norm(a.flatten())
        norm_b = np.linalg.norm(b.flatten())
        return float(dot_product / (norm_a * norm_b + 1e-8))
    
    def euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance-based similarity between two vectors.
        Returns 1 / (1 + distance) so higher values indicate more similarity."""
        distance = np.linalg.norm(a.flatten() - b.flatten())
        return float(1.0 / (1.0 + distance))
    
    def captum_sensitivity_analysis(self, X: np.ndarray, explanations: np.ndarray, index: int, explainer_func=None) -> float:
        """
        Compute sensitivity using Captum's sensitivity_max with the SAME explainer function
        that generated the input explanations for fair comparison.
        """
        if not CAPTUM_AVAILABLE:
            return self._manual_sensitivity_analysis(X, explanations, index)
        
        try:
            import torch
            import torch.nn as nn
            
            x_i = X[index]
            
            # Convert to tensor
            input_tensor = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)
            input_tensor.requires_grad_(True)
            
            # If no explainer_func provided, compute explanation variance directly
            if explainer_func is None:
                print("Warning: No explainer function provided, using explanation variance as sensitivity")
                return self._explanation_variance_sensitivity(X, explanations, index)
            
            # Use Captum's sensitivity_max with the SAME explainer function
            sensitivity = sensitivity_max(
                explanation_func=explainer_func,  # Same explainer that generated explanations!
                inputs=input_tensor,
                perturb_radius=self.noise_level,
                n_perturb_samples=self.n_samples,
                target=0
            )
            
            # Handle NaN/inf values
            if torch.isnan(sensitivity) or torch.isinf(sensitivity):
                print(f"Warning: Captum sensitivity returned {sensitivity.item()}, using fallback")
                return self._explanation_variance_sensitivity(X, explanations, index)
            
            return float(sensitivity.item())
            
        except Exception as e:
            print(f"Captum sensitivity failed: {e}, using fallback")
            return self._explanation_variance_sensitivity(X, explanations, index)
    
    def _explanation_variance_sensitivity(self, X: np.ndarray, explanations: np.ndarray, index: int) -> float:
        """Compute sensitivity as variance in explanation values."""
        expl_i = explanations[index]
        
        # Compute explanation variance as sensitivity measure
        # Higher variance = less stable = higher sensitivity
        variance = np.var(expl_i)
        
        # Normalize to [0, 1] range approximately
        normalized_sensitivity = min(1.0, variance / (np.mean(np.abs(expl_i)) + 1e-8))
        
        return float(normalized_sensitivity)
    
    def _manual_sensitivity_analysis(self, X: np.ndarray, explanations: np.ndarray, index: int) -> float:
        """Fallback sensitivity analysis when Captum is not available."""
        x_i = X[index]
        expl_i = explanations[index]
        
        # Compute sensitivity as stability of explanation under small perturbations
        sensitivities = []
        for _ in range(self.n_samples):
            # Add small noise to input
            noise = np.random.normal(0, self.noise_level, x_i.shape)
            x_perturbed = x_i + noise
            
            # Since we don't have a model, approximate explanation change
            # Use gradient approximation based on current explanation
            gradient_approx = expl_i * (noise / (self.noise_level + 1e-8))
            expl_change = np.linalg.norm(gradient_approx)
            sensitivities.append(expl_change)
        
        # Return mean sensitivity (higher = less stable)
        return float(np.mean(sensitivities))
    
    def knn_similarity_robustness(self, X: np.ndarray, explanations: np.ndarray, index: int, y: np.ndarray = None) -> float:
        """
        K-nearest neighbor robustness using average similarity between explanations.
        Higher values indicate more robust (similar) explanations.
        
        Args:
            X: Input features
            explanations: Explanation vectors
            index: Index of the sample to analyze
            y: Class labels (required when same_class_only=True)
        """
        x_i = X[index]
        expl_i = explanations[index]
        
        # Compute distances to find k nearest neighbors
        distances = []
        for j in range(len(X)):
            if j != index:
                # If same_class_only is enabled, only consider same-class neighbors
                if self.same_class_only:
                    if y is None:
                        raise ValueError("Class labels (y) must be provided when same_class_only=True")
                    if y[j] != y[index]:
                        continue  # Skip different class neighbors
                
                dist = np.linalg.norm(x_i - X[j])
                distances.append((dist, j))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(self.k, len(distances))]
        
        if len(k_nearest) == 0:
            return 0.0
        
        # Compute average similarity with k nearest neighbors using specified metric
        similarities = []
        for _, neighbor_idx in k_nearest:
            if self.similarity_metric == "cosine":
                similarity = self.cosine_similarity(expl_i, explanations[neighbor_idx])
            elif self.similarity_metric == "euclidean":
                similarity = self.euclidean_similarity(expl_i, explanations[neighbor_idx])
            else:
                raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
            similarities.append(similarity)
        
        return float(np.mean(similarities))


def _load_split(root: Path, dataset: str, split: str = "test"):
    """Load dataset split."""
    try:
        import torch
        X = torch.load(root / "data" / dataset / f"X{split}.pt", map_location="cpu")
        y = torch.load(root / "data" / dataset / f"y{split}.pt", map_location="cpu")
        
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
        raise RuntimeError("PyTorch is required for loading datasets")


def _load_explanations(root: Path, dataset: str, expl_path: str, X: np.ndarray):
    """Load explanations, falling back to X if not available."""
    if expl_path:
        expl_file = Path(expl_path)
    else:
        expl_file = root / "explanations" / "results_medoid" / f"{dataset}_attributions.npy"
    
    if expl_file.exists():
        try:
            expl = np.load(expl_file, allow_pickle=True)
            if hasattr(expl, 'item') and isinstance(expl.item(), dict):
                # Handle different explanation formats
                expl_dict = expl.item()
                if 'attributions' in expl_dict:
                    expl = expl_dict['attributions']
                elif 'gradients' in expl_dict:
                    expl = expl_dict['gradients']
                else:
                    # Use first available array
                    expl = next(iter(expl_dict.values()))
            
            if expl.shape[0] != X.shape[0]:
                print(f"Warning: explanation shape {expl.shape} doesn't match X shape {X.shape}")
                print("Falling back to using X as explanations")
                expl = X.copy()
                expl_src = "X (fallback)"
            else:
                expl_src = str(expl_file)
        except Exception as e:
            print(f"Warning: failed to load explanations from {expl_file}: {e}")
            print("Falling back to using X as explanations")
            expl = X.copy()
            expl_src = "X (fallback)"
    else:
        print(f"Explanations file {expl_file} not found. Using X as explanations.")
        expl = X.copy()
        expl_src = "X (fallback)"
    
    return expl.astype(float), expl_src


def run_correlation_experiment(
    dataset: str,
    split: str = "test",
    n_samples: int = 200,
    epsilon: float = 0.1,
    k: int = 5,
    seed: int = 42,
    expl_path: str = "",
    sim_metric: str = "gower",
    explainer_method: str = "ig",  # New parameter for consistent explainer
    same_class_only: bool = False,  # New parameter for same-class k-NN
    similarity_metric: str = "euclidean",  # New parameter for k-NN similarity calculation
    output_path: str = ""
) -> pd.DataFrame:
    """
    Run the main correlation experiment using consistent explainer across all metrics.
    """
    print(f"[case_align_correlation] Starting Consistent Explainer Experiment:")
    print(f"  dataset={dataset}, split={split}, n_samples={n_samples}")
    print(f"  explainer_method={explainer_method}, noise_level={epsilon}, k={k}, seed={seed}")
    print(f"  sim_metric={sim_metric}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Load data and generate explanations using specified method
    X, y = _load_split(ROOT, dataset, split)
    
    # Generate explanations using the specified explainer method
    print(f"[case_align_correlation] Generating {explainer_method.upper()} explanations...")
    try:
        from explainers.captum_explain import explain_batch
        from load.load_net import load_net
        import torch
        import os
        
        # Load trained model with weights - assume model1 for now
        model_name = "model1"
        
        # Load model architecture
        net_module = load_net(dataset)
        model = net_module.recover_net("smallNN")  # Use smallNN by default
        
        # Load trained weights
        model_path = ROOT.parent / "models" / dataset / f"{dataset}_{model_name}.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"[case_align_correlation] ✓ Loaded trained model: {model_path}")
        else:
            print(f"[case_align_correlation] Warning: Model weights not found at {model_path}, using random weights")
        
        model.eval()
        
        # Generate explanations using specified method
        import torch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        explanations_dict = explain_batch(model, X_tensor, methods=[explainer_method], batch_size=64)
        expl = explanations_dict[explainer_method].detach().cpu().numpy()
        expl_src = f"{explainer_method.upper()} generated"
        
        # Create explainer function for sensitivity analysis
        if explainer_method == "ig":
            from captum.attr import IntegratedGradients
            explainer_func = IntegratedGradients(model).attribute
        elif explainer_method == "dl":
            from captum.attr import DeepLift 
            explainer_func = DeepLift(model).attribute
        elif explainer_method == "lrp":
            from explainers.lrp import LRP
            explainer_func = LRP(model).attribute
        else:
            raise ValueError(f"Unknown explainer method: {explainer_method}")
            
        print(f"[case_align_correlation] ✓ Generated {explainer_method.upper()} explanations")
        
    except Exception as e:
        print(f"[case_align_correlation] Warning: Failed to generate {explainer_method} explanations: {e}")
        print(f"[case_align_correlation] Falling back to loading from file or using X...")
        expl, expl_src = _load_explanations(ROOT, dataset, expl_path, X)
        explainer_func = None  # Will use Saliency fallback
    
    print(f"[case_align_correlation] Loaded data:")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  explanations source: {expl_src}, shape: {expl.shape}")
    
    # Sample indices for evaluation
    rng = np.random.default_rng(seed)
    n_eval = min(n_samples, X.shape[0])
    sample_indices = rng.choice(X.shape[0], size=n_eval, replace=False)
    print(f"[case_align_correlation] Evaluating on {n_eval} samples")
    
    # Initialize case align with like-only variant
    case_align = RobustnessCBR(
        k=k,
        m_unlike=1,  # Not used in like-only mode
        sim_metric=sim_metric,
        problem_metric=sim_metric,
        like_only=True,  # This is the key parameter for like-only variant
        robust_mode="geom",
        random_state=seed
    )
    case_align.fit(X, y, expl)
    print("[case_align_correlation] ✓ Case align fitted successfully")
    
    # Initialize Captum-based metrics
    captum_metrics = CaptumSensitivityMetrics(k=k, noise_level=epsilon, n_samples=10, same_class_only=same_class_only, similarity_metric=similarity_metric)
    print(f"[case_align_correlation] ✓ Captum metrics initialized (Captum available: {CAPTUM_AVAILABLE}, same_class_only: {same_class_only}, similarity_metric: {similarity_metric})")
    
    # Compute metrics for all samples
    results = []
    
    for i, sample_idx in enumerate(sample_indices):
        if (i + 1) % 50 == 0 or i + 1 == len(sample_indices):
            print(f"[case_align_correlation] Progress: {i + 1}/{len(sample_indices)}")
        
        # Compute case align metrics
        ca_result = case_align.compute_for_index(sample_idx)
        
        # Compute Captum-based robustness metrics using SAME explainer
        sensitivity = captum_metrics.captum_sensitivity_analysis(X, expl, sample_idx, explainer_func)
        knn_similarity = captum_metrics.knn_similarity_robustness(X, expl, sample_idx, y)
        
        # Also compute same-class-only k-NN similarity if enabled
        if captum_metrics.same_class_only:
            knn_same_class = captum_metrics.knn_similarity_robustness(X, expl, sample_idx, y)
        else:
            knn_same_class = knn_similarity  # Same as regular k-NN when not filtering by class
        
        results.append({
            'index': sample_idx,
            'class': y[sample_idx],
            # Case align metrics (like-only variant)
            'case_align_S_plus': ca_result.S_plus,
            'case_align_R_bounded': ca_result.R_bounded,
            'case_align_k_like': ca_result.k_like_x,
            # Traditional robustness metrics
            'captum_sensitivity': sensitivity,
            'knn_similarity_robustness': knn_similarity,
            'knn_similarity_same_class': knn_same_class,
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    print(f"[case_align_correlation] ✓ Metrics computed successfully")
    
    # Compute correlations
    case_align_cols = ['case_align_S_plus', 'case_align_R_bounded']
    traditional_cols = ['captum_sensitivity', 'knn_similarity_robustness', 'knn_similarity_same_class']
    
    print(f"\n[case_align_correlation] Computing correlations...")
    correlation_results = []
    
    for ca_col in case_align_cols:
        for trad_col in traditional_cols:
            # Remove any infinite or NaN values for correlation computation
            mask = np.isfinite(df[ca_col]) & np.isfinite(df[trad_col])
            if mask.sum() < 3:
                continue
                
            ca_vals = df[ca_col][mask]
            trad_vals = df[trad_col][mask]
            
            # Compute Pearson and Spearman correlations
            try:
                pearson_r, pearson_p = pearsonr(ca_vals, trad_vals)
                spearman_r, spearman_p = spearmanr(ca_vals, trad_vals)
                
                correlation_results.append({
                    'case_align_metric': ca_col,
                    'traditional_metric': trad_col,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_samples': mask.sum()
                })
            except Exception as e:
                print(f"Warning: correlation failed for {ca_col} vs {trad_col}: {e}")
    
    # Display results
    print(f"\n[case_align_correlation] CORRELATION RESULTS:")
    print("=" * 80)
    
    corr_df = pd.DataFrame(correlation_results)
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            print(f"{row['case_align_metric']} vs {row['traditional_metric']}:")
            print(f"  Pearson r={row['pearson_r']:.3f} (p={row['pearson_p']:.3f})")
            print(f"  Spearman r={row['spearman_r']:.3f} (p={row['spearman_p']:.3f})")
            print(f"  n={row['n_samples']}")
            print()
        
        print(f"\n[case_align_correlation] INTERPRETATION:")
        print(f"Expected relationships:")
        print(f"  - Case Align ↑ & Captum Sensitivity ↓ = Both indicate stability (NEGATIVE correlation expected)")
        print(f"  - Case Align ↑ & k-NN Similarity ↑ = Both indicate consistency (POSITIVE correlation expected)")
    
    # Summary statistics 
    print(f"\n[case_align_correlation] SUMMARY STATISTICS:")
    print("=" * 50)
    print(df[case_align_cols + traditional_cols].describe())
    
    # Save results if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_file, index=False)
        print(f"\n[case_align_correlation] Detailed results saved to: {output_file}")
        
        # Save correlation summary
        corr_file = output_file.parent / f"{output_file.stem}_correlations.csv"
        if not corr_df.empty:
            corr_df.to_csv(corr_file, index=False)
            print(f"[case_align_correlation] Correlation results saved to: {corr_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Case Align vs Captum Sensitivity Correlation Experiment"
    )
    parser.add_argument("--dataset", type=str, default="adult", 
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="test", 
                       help="Data split to use")
    parser.add_argument("--n_samples", type=int, default=200, 
                       help="Number of samples to evaluate")
    parser.add_argument("--epsilon", type=float, default=0.1, 
                       help="Noise level for sensitivity analysis")
    parser.add_argument("--k", type=int, default=5, 
                       help="Number of neighbors for case align and k-NN similarity")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--expl_path", type=str, default="", 
                       help="Path to explanations file")
    parser.add_argument("--sim_metric", type=str, default="gower", 
                       choices=["gower", "cosine", "spearman"],
                       help="Similarity metric for case align")
    parser.add_argument("--explainer_method", type=str, default="ig",
                       choices=["ig", "dl", "lrp"],
                       help="Explainer method (ig=IntegratedGradients, dl=DeepLift, lrp=LRP)")
    parser.add_argument("--same_class_only", action="store_true",
                       help="Use only same-class neighbors for k-NN similarity robustness")
    parser.add_argument("--similarity_metric", type=str, default="euclidean",
                       choices=["euclidean", "cosine"],
                       help="Similarity metric for k-NN similarity calculation (euclidean or cosine)")
    parser.add_argument("--output", type=str, default="", 
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        output_dir = ROOT / "results" / "case_align_consistent_correlation"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{args.dataset}_{args.split}_{args.explainer_method}_correlation_results.csv")
    
    # Run experiment
    df = run_correlation_experiment(
        dataset=args.dataset,
        split=args.split,
        n_samples=args.n_samples,
        epsilon=args.epsilon,
        k=args.k,
        seed=args.seed,
        expl_path=args.expl_path,
        sim_metric=args.sim_metric,
        explainer_method=args.explainer_method,
        same_class_only=args.same_class_only,
        similarity_metric=args.similarity_metric,
        output_path=args.output
    )
    
    print(f"\n[case_align_correlation] Consistent Explainer Experiment completed successfully!")
    print(f"Results shape: {df.shape}")
    print(f"Explainer method: {args.explainer_method.upper()}")
    print(f"Metrics analyzed: Case Align (like-only) vs {args.explainer_method.upper()} Sensitivity & k-NN Similarity")


if __name__ == "__main__":
    main()