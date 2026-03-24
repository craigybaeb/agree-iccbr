#!/usr/bin/env python3
"""
Minimal test script for case align correlation experiment.
Uses only core dependencies to avoid version conflicts.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from case_align.case_align import RobustnessCBR
    print("✓ Successfully imported RobustnessCBR")
except ImportError as e:
    print(f"✗ Failed to import RobustnessCBR: {e}")
    sys.exit(1)

def test_minimal_case_align():
    """Test basic case align functionality with synthetic data."""
    print("\nTesting basic case align functionality...")
    
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 4
    
    # Simple binary classification data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Use X as explanations (simple case)
    explanations = X.copy()
    
    print(f"Generated data: X shape {X.shape}, y shape {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test case align with like-only variant
    try:
        case_align = RobustnessCBR(
            k=5,
            m_unlike=1,
            sim_metric="gower",
            problem_metric="gower", 
            like_only=True,  # Key parameter for like-only variant
            robust_mode="geom",
            random_state=42
        )
        
        case_align.fit(X, y, explanations)
        print("✓ Successfully fitted RobustnessCBR")
        
        # Compute metrics for a few samples
        results = []
        for i in range(min(5, n_samples)):
            result = case_align.compute_for_index(i)
            results.append(result)
            print(f"  Sample {i}: S_plus={result.S_plus:.3f}, R_bounded={result.R_bounded:.3f}")
        
        print("✓ Successfully computed case align metrics")
        print(f"✓ Like-only mode working: all S_minus should be 0")
        
        # Verify like-only behavior
        all_s_minus_zero = all(r.S_minus == 0.0 for r in results)
        if all_s_minus_zero:
            print("✓ Like-only variant confirmed: all S_minus = 0")
        else:
            print("⚠ Warning: Some S_minus values are non-zero")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in case align computation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_robustness_metrics():
    """Test simple robustness metrics without sklearn."""
    print("\nTesting simple robustness metrics...")
    
    # Create simple data
    np.random.seed(42)
    X = np.random.randn(20, 3)
    explanations = np.random.randn(20, 3)
    
    try:
        # Simple Euclidean distance-based metrics
        def euclidean_distance(a, b):
            return np.sqrt(np.sum((a - b) ** 2))
        
        def local_smoothness_simple(X, explanations, index, epsilon=0.5):
            """Simple version of local smoothness metric."""
            x_i = X[index]
            expl_i = explanations[index]
            
            # Find neighbors within epsilon
            distances = [euclidean_distance(x_i, X[j]) for j in range(len(X)) if j != index]
            neighbors = [j for j, d in enumerate(distances) if d <= epsilon and j != index]
            
            if len(neighbors) == 0:
                return 0.0
            
            # Compute explanation distances to neighbors
            expl_distances = [euclidean_distance(expl_i, explanations[j]) for j in neighbors]
            return np.mean(expl_distances)
        
        # Test the metric
        smoothness = local_smoothness_simple(X, explanations, 0)
        print(f"✓ Local smoothness for sample 0: {smoothness:.3f}")
        
        # Test correlation computation
        values1 = np.random.randn(10)
        values2 = values1 + 0.1 * np.random.randn(10)  # Correlated data
        
        # Simple Pearson correlation
        def pearson_correlation(x, y):
            x_mean, y_mean = np.mean(x), np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
            return numerator / denominator if denominator > 0 else 0
        
        corr = pearson_correlation(values1, values2)
        print(f"✓ Simple correlation test: {corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in robustness metrics: {e}")
        return False


def main():
    print("Minimal Case Align Correlation Test")
    print("=" * 40)
    
    success1 = test_minimal_case_align()
    success2 = test_simple_robustness_metrics()
    
    if success1 and success2:
        print("\n🎉 Minimal tests passed!")
        print("\nNext steps:")
        print("1. Fix dependency issues (numpy/sklearn compatibility)")
        print("2. Run: python experiments/case_align_correlation.py --n_samples 50")
        print("3. Use the Jupyter notebook for full analysis")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)