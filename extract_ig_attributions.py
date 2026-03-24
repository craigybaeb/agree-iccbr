#!/usr/bin/env python3
"""
Script to extract IG attributions and save them in the format expected by the visualization notebook.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(ROOT))

from explainers.captum_explain import explain_dataset

def extract_and_convert_attributions(
    dataset: str = "adult",
    model_name: str = "model1", 
    split: str = "test"
):
    """
    Extract IG attributions using captum_explain.py and convert to expected format.
    """
    print(f"Extracting IG attributions for {dataset} {model_name} {split}...")
    
    # Extract using existing captum_explain function
    # This saves to: explainers/{dataset}/{dataset}_{model_name}_{split}_ig.pt
    results = explain_dataset(
        dataset=dataset,
        model_name=model_name,
        split=split,
        methods=["ig"],  # Only IG
        batch_size=256
    )
    
    ig_file = Path(results["ig"])
    print(f"Generated IG file: {ig_file}")
    
    # Load the PyTorch tensor
    attributions = torch.load(ig_file, map_location="cpu").numpy()
    print(f"Attribution shape: {attributions.shape}")
    
    # Create output directory in expected format
    output_dir = ROOT / "explanations" / "results_medoid" / f"{dataset}_{model_name}" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array in expected location
    output_file = output_dir / "attributions.npy"
    np.save(output_file, attributions)
    
    print(f"✅ Saved attributions to: {output_file}")
    print(f"   Shape: {attributions.shape}")
    print(f"   Ready for visualization notebook!")
    
    return output_file

if __name__ == "__main__":
    # Extract for adult dataset test split
    extract_and_convert_attributions("adult", "model1", "test")
    
    # Optionally extract for other splits
    # extract_and_convert_attributions("adult", "model1", "train")
    # extract_and_convert_attributions("adult", "model1", "val")