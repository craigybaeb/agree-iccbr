# Extract IG Attributions - Add this cell to any notebook
print("Extracting IG attributions...")

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path if needed
current_dir = Path.cwd()
if current_dir.name == 'experiments':
    root_dir = current_dir.parent
else:
    root_dir = current_dir
    
sys.path.append(str(root_dir))

from explainers.captum_explain import explain_batch, _load_model, _load_dataset, _baseline_for_dataset

# Configuration
dataset = "adult"
model_name = "model1" 
split = "test"

# Load data and model
X_test, _ = _load_dataset(root_dir, dataset, split=split)
model = _load_model(root_dir, dataset, model_name)
baseline = _baseline_for_dataset(root_dir, dataset, X_test)

print(f"Data shape: {X_test.shape}")
print(f"Model loaded: {model_name}")

# Extract IG attributions
attrs = explain_batch(
    model=model, 
    X=X_test, 
    methods=["ig"],  # Only Integrated Gradients
    baselines=baseline,
    batch_size=256
)

ig_attributions = attrs["ig"].numpy()
print(f"IG attributions shape: {ig_attributions.shape}")

# Save in expected format
output_dir = root_dir / "explanations" / "results_medoid" / f"{dataset}_{model_name}" / split
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "attributions.npy"
np.save(output_file, ig_attributions)

print(f"✅ Saved IG attributions to: {output_file}")
print("Ready for case align analysis!")