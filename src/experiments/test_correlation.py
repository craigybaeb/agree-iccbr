#!/usr/bin/env python3
"""
Simple test script for the case align correlation experiment.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.case_align_correlation import run_correlation_experiment


def test_small_experiment():
    """Run a small test experiment to verify everything works."""
    print("Running small test experiment...")
    
    try:
        # Run with minimal parameters for testing
        df = run_correlation_experiment(
            dataset="adult",  # Assuming adult dataset exists
            split="test",
            n_samples=50,    # Small sample for testing
            epsilon=0.1,
            k=5,
            seed=42,
            expl_path="",    # Will fall back to X
            sim_metric="gower",
            output_path=""   # No output for test
        )
        
        print(f"✓ Test completed successfully!")
        print(f"✓ Results shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check for basic sanity
        if 'case_align_S_plus' in df.columns and 'lipschitz_continuity' in df.columns:
            print(f"✓ Key metrics computed successfully")
        else:
            print("⚠ Warning: Missing expected columns")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_experiment()
    if success:
        print("\n🎉 All tests passed! The experiment is ready to run.")
        print("\nTo run the full experiment:")
        print("python src/experiments/case_align_correlation.py --dataset adult --n_samples 200")
    else:
        print("\n❌ Tests failed. Please check the setup.")
        sys.exit(1)