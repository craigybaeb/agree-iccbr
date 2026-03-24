#!/usr/bin/env python3
"""
Case Align Correlation Analysis Module

This module contains the main analysis classes and functions for running 
correlation experiments between Case Align and traditional robustness metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import case align
from case_align.case_align import RobustnessCBR

# Import Captum for sensitivity analysis
try:
    import torch
    from captum.attr import IntegratedGradients, Saliency
    from captum.metrics import sensitivity_max
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class RealCaptumSensitivityMetrics:
    """
    Robustness metrics using the ACTUAL trained model and explainer functions.
    This ensures meaningful correlation analysis with real models.
    """
    
    def __init__(self, model, explainer, k: int = 5, noise_level: float = 0.1, n_samples: int = 10):
        self.model = model
        self.explainer = explainer
        self.k = k
        self.noise_level = noise_level
        self.n_samples = n_samples
        
        # Ensure model is in eval mode
        if hasattr(model, 'eval'):
            model.eval()
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a.flatten(), b.flatten())
        norm_a = np.linalg.norm(a.flatten())
        norm_b = np.linalg.norm(b.flatten())
        return float(dot_product / (norm_a * norm_b + 1e-8))
    
    def captum_sensitivity_analysis(self, X: np.ndarray, index: int) -> float:
        """
        Compute sensitivity using manual perturbation analysis with the ACTUAL
        model and explainer that were used to generate the explanations.
        """
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum is not available. Install captum to use sensitivity analysis.")
        
        try:
            import torch
            
            x_i = X[index]  # This is a 1D numpy array
            
            # Convert to tensor with proper shape (add batch dimension)
            input_tensor = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)  # Shape: [1, features]
            input_tensor.requires_grad_(True)
            
            # Get original explanation
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            original_attr = self.explainer.attribute(input_tensor, target=predicted_class).detach().numpy().flatten()
            
            # Generate perturbed explanations
            sensitivities = []
            for i in range(self.n_samples):
                # Add noise to input (keep as 1D)
                noise = np.random.normal(0, self.noise_level, x_i.shape)
                x_perturbed = x_i + noise
                
                # Convert to tensor with proper shape (add batch dimension)
                perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32).unsqueeze(0)  # Shape: [1, features]
                perturbed_tensor.requires_grad_(True)
                
                # Get perturbed explanation
                with torch.no_grad():
                    outputs = self.model(perturbed_tensor)
                    predicted_class_pert = torch.argmax(outputs, dim=1).item()
                
                perturbed_attr = self.explainer.attribute(perturbed_tensor, target=predicted_class_pert).detach().numpy().flatten()
                
                # Compute sensitivity as the L2 norm of the difference
                sensitivity = np.linalg.norm(original_attr - perturbed_attr)
                sensitivities.append(sensitivity)
            
            final_sensitivity = float(np.mean(sensitivities))
            
            return final_sensitivity
            
        except Exception as e:
            raise RuntimeError(f"Captum sensitivity analysis failed for sample {index}: {e}")

    def knn_similarity_robustness(self, X: np.ndarray, explanations: np.ndarray, index: int) -> float:
        """
        K-nearest neighbor robustness using cosine similarity of explanations.
        Higher values indicate more robust (similar) explanations.
        """
        x_i = X[index]
        expl_i = explanations[index]
        
        # Find k-nearest neighbors in INPUT SPACE
        distances = []
        for j in range(len(X)):
            if j != index:
                dist = np.linalg.norm(x_i - X[j])
                distances.append((dist, j))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(self.k, len(distances))]
        
        if len(k_nearest) == 0:
            return 0.0
        
        # Compute cosine similarity of EXPLANATIONS with k-nearest neighbors
        similarities = []
        for _, neighbor_idx in k_nearest:
            similarity = self.cosine_similarity(expl_i, explanations[neighbor_idx])
            similarities.append(similarity)
        
        return float(np.mean(similarities))


def load_model_and_explainer(dataset: str = "adult", model_name: str = "model1"):
    """Load the actual trained model and create explainer."""
    import torch
    from captum.attr import IntegratedGradients, Saliency
    from load.load_net import load_net
    
    # First, determine the correct input dimensions by loading the data
    try:
        data_dir = ROOT / "data" / dataset
        X_sample = torch.load(data_dir / "Xtest.pt", map_location="cpu")
        n_features = X_sample.shape[1]
        print(f"✓ Detected {n_features} features for {dataset} dataset")
    except Exception as e:
        raise RuntimeError(f"Failed to determine input dimensions for {dataset}: {e}")
    
    # Load the actual trained model
    try:
        # Load model architecture
        net_module = load_net(dataset)
        model = net_module.recover_net("smallNN")
        
        # Load trained weights
        model_path = ROOT.parent / "models" / dataset / f"{dataset}_{model_name}.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"✓ Loaded trained model: {model_path}")
        else:
            print(f"⚠ Model weights not found at {model_path}, using random weights")
        
        model.eval()
        
        # Test model with the correct input dimensions
        dummy_input = torch.randn(1, n_features)
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"✓ Model test successful - output shape: {test_output.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {dataset}_{model_name}: {e}")
    
    # Create explainer
    try:
        explainer = IntegratedGradients(model)
        print("✓ IntegratedGradients explainer created")
    except Exception as e:
        raise RuntimeError(f"Failed to create IntegratedGradients explainer: {e}")
    
    return model, explainer


def load_data(dataset: str = "adult", split: str = "test"):
    """Load dataset."""
    try:
        import torch
        data_dir = ROOT / "data" / dataset
        X = torch.load(data_dir / f"X{split}.pt", map_location="cpu").numpy()
        y = torch.load(data_dir / f"y{split}.pt", map_location="cpu").numpy()
        
        if y.ndim == 2 and y.shape[1] > 1:
            y = y.argmax(axis=1)
        y = y.reshape(-1)
        
        print(f"✓ Loaded data: X shape {X.shape}, y shape {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Using synthetic data...")
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        print(f"Generated synthetic data: X shape {X.shape}, y shape {y.shape}")
        return X, y


def generate_explanations(model, explainer, X: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """Generate explanations using the model and explainer."""
    print("Generating explanations with the actual model and explainer...")
    explanations = []
    X_tensor = torch.tensor(X[sample_indices], dtype=torch.float32)
    
    explanation_failures = 0
    for i, x in enumerate(X_tensor):
        x_single = x.unsqueeze(0).requires_grad_(True)
        try:
            # Get predicted class for target
            with torch.no_grad():
                outputs = model(x_single)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Generate attribution with proper target
            attr = explainer.attribute(x_single, target=predicted_class).detach().numpy().flatten()
            explanations.append(attr)
        except Exception as e:
            explanation_failures += 1
            if explanation_failures <= 3:
                print(f"Warning: explanation failed for sample {i}: {e}")
            raise RuntimeError(f"Explanation generation failed for sample {i}: {e}")
        
        if (i + 1) % 20 == 0:
            print(f"  Explanations progress: {i + 1}/{len(X_tensor)} (failures: {explanation_failures})")
    
    explanations = np.array(explanations)
    if explanation_failures > 0:
        print(f"⚠ Total explanation failures: {explanation_failures}/{len(X_tensor)}")
    print(f"✓ Generated explanations: shape {explanations.shape}")
    
    return explanations


def run_correlation_experiment(
    dataset: str = "adult",
    split: str = "test",
    n_samples: int = 100,
    k: int = 5,
    noise_level: float = 0.1,
    seed: int = 42,
    model_name: str = "model1",
    sim_metric: str = "gower"
) -> pd.DataFrame:
    """
    Main function to run correlation experiment using actual trained model and explainer.
    """
    print(f"🔬 Case Align vs Captum Sensitivity Correlation Experiment")
    print(f"Dataset: {dataset}, Model: {model_name}, Samples: {n_samples}")
    print(f"Noise level: {noise_level}, K: {k}, Similarity: {sim_metric}")
    print("-" * 60)
    
    # Set random seed
    np.random.seed(seed)
    
    # Load components
    print("1. Loading model and explainer...")
    model, explainer = load_model_and_explainer(dataset, model_name)
    
    print("2. Loading data...")
    X, y = load_data(dataset, split)
    
    print("3. Sampling data...")
    n_eval = min(n_samples, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], size=n_eval, replace=False)
    print(f"✓ Evaluating {n_eval} samples")
    
    print("4. Generating explanations...")
    explanations = generate_explanations(model, explainer, X, sample_indices)
    
    print("5. Setting up Case Align...")
    case_align = RobustnessCBR(
        k=k,
        m_unlike=1,
        sim_metric=sim_metric,
        problem_metric=sim_metric,
        like_only=True,
        robust_mode="geom",
        random_state=seed
    )
    case_align.fit(X[sample_indices], y[sample_indices], explanations)
    print("✓ Case align fitted with actual explanations")
    
    print("6. Setting up sensitivity metrics...")
    sensitivity_metrics = RealCaptumSensitivityMetrics(
        model=model,
        explainer=explainer,
        k=k, 
        noise_level=noise_level, 
        n_samples=10
    )
    
    print("7. Computing correlations...")
    results = []
    
    for i, sample_idx_in_subset in enumerate(range(n_eval)):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_eval}")
        
        # Case align metrics
        ca_result = case_align.compute_for_index(sample_idx_in_subset)
        
        # Real Captum sensitivity analysis
        original_idx = sample_indices[sample_idx_in_subset]
        sensitivity = sensitivity_metrics.captum_sensitivity_analysis(X, original_idx)
        
        # k-NN similarity robustness  
        knn_similarity = sensitivity_metrics.knn_similarity_robustness(
            X[sample_indices], explanations, sample_idx_in_subset
        )
        
        results.append({
            'index': original_idx,
            'class': y[original_idx],
            'case_align_S_plus': ca_result.S_plus,
            'case_align_R_bounded': ca_result.R_bounded,
            'captum_sensitivity': sensitivity,
            'knn_similarity_robustness': knn_similarity,
        })
    
    print("✓ All metrics computed successfully")
    
    return pd.DataFrame(results)


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlations between Case Align and traditional metrics."""
    from scipy.stats import pearsonr, spearmanr
    
    case_align_cols = ['case_align_S_plus', 'case_align_R_bounded']
    traditional_cols = ['captum_sensitivity', 'knn_similarity_robustness']
    
    results = []
    
    for ca_col in case_align_cols:
        for trad_col in traditional_cols:
            # Remove infinite and NaN values
            mask = np.isfinite(df[ca_col]) & np.isfinite(df[trad_col])
            if mask.sum() < 3:
                continue
                
            ca_vals = df[ca_col][mask]
            trad_vals = df[trad_col][mask]
            
            try:
                pearson_r, pearson_p = pearsonr(ca_vals, trad_vals)
                spearman_r, spearman_p = spearmanr(ca_vals, trad_vals)
                
                results.append({
                    'case_align_metric': ca_col.replace('case_align_', ''),
                    'traditional_metric': trad_col,
                    'pearson_r': pearson_r,
                    'spearman_r': spearman_r,
                    'pearson_p': pearson_p,
                    'spearman_p': spearman_p,
                    'n_samples': mask.sum()
                })
                
            except Exception as e:
                print(f"Warning: correlation failed for {ca_col} vs {trad_col}: {e}")
    
    return pd.DataFrame(results)


def diagnose_correlation_issues(df: pd.DataFrame) -> Dict[str, any]:
    """Diagnose potential issues with correlations."""
    diagnosis = {
        'low_variance_metrics': [],
        'constant_metrics': [],
        'high_nan_count': [],
        'recommendations': []
    }
    
    metrics = ['case_align_S_plus', 'case_align_R_bounded', 'captum_sensitivity', 'knn_similarity_robustness']
    
    for metric in metrics:
        if metric in df.columns:
            finite_vals = df[metric][np.isfinite(df[metric])]
            
            if len(finite_vals) == 0:
                diagnosis['high_nan_count'].append(metric)
                continue
                
            variance = finite_vals.var()
            
            if variance < 1e-6:
                diagnosis['constant_metrics'].append(metric)
            elif variance < 1e-4:
                diagnosis['low_variance_metrics'].append(metric)
    
    # Generate recommendations
    if 'captum_sensitivity' in diagnosis['constant_metrics']:
        diagnosis['recommendations'].append("Increase noise level (epsilon) from 0.1 to 0.3-0.5 for better sensitivity variance")
    
    if diagnosis['low_variance_metrics']:
        diagnosis['recommendations'].append("Consider standardizing metrics before correlation analysis")
        
    if diagnosis['high_nan_count']:
        diagnosis['recommendations'].append("Check for numerical stability issues in metric computation")
    
    return diagnosis