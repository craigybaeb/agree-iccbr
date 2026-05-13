"""
SmoothGrad Parameter Tuning for MNIST
======================================
Systematically explores nt_samples and nt_stdevs combinations on a small set of MNIST instances.
Computes attributions and metrics (Case Align, Sensitivity) for each parameter combination.
Visualizes both raw and smoothed attributions to identify visually smooth + robust solutions.
"""
import sys
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from captum.attr import IntegratedGradients, DeepLift, NoiseTunnel
from captum.metrics import sensitivity_max
from scipy.stats import entropy

# ============================================================================
# Path setup
# ============================================================================
current_dir = Path.cwd()
if current_dir.name == "experiments":
    SRC_DIR = current_dir.parent
elif (current_dir / "src").exists():
    SRC_DIR = current_dir / "src"
else:
    SRC_DIR = current_dir

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from case_align.metrics import rankdata, safe_normalise_rows
from explainers.lrp import LRP
from train_mnist_model import MNISTNet

# ============================================================================
# Constants
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = SRC_DIR / "models" / "mnist" / "mnist_best_model.pt"
DATA_DIR = SRC_DIR / "data"
OUTPUT_DIR = SRC_DIR / "results" / "smoothgrad_param_tuning"
N_INSTANCES = 10  # Small sample for interactive tuning
K_CASE_ALIGN = 5
SIM_METRIC = "gower"
PERTURB_RADIUS = 0.1
N_PERTURB = 10

# Parameter grid to sweep
NT_SAMPLES_GRID = [4, 8, 16, 32]
NT_STDEVS_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ============================================================================
# Model & Data Loading
# ============================================================================
def load_mnist_model():
    """Load MNIST model."""
    model = MNISTNet().to(DEVICE)
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle both direct state_dict and wrapped checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model.eval()
    return model

def load_mnist_data(n_samples=N_INSTANCES):
    """Load MNIST test set and select n_samples random instances."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
    indices = np.random.choice(len(test_set), size=n_samples, replace=False)
    images = torch.stack([test_set[i][0] for i in indices]).to(DEVICE)
    labels = torch.tensor([test_set[i][1] for i in indices]).to(DEVICE)
    return images, labels

# ============================================================================
# Attribution & Metrics
# ============================================================================
def compute_attribution(model, image, method="ig", nt_samples=None, nt_stdevs=0.1):
    """
    Compute attribution for a single image.
    
    Args:
        model: Neural network model
        image: Input image tensor (1, 1, 28, 28)
        method: "ig", "deeplift", or "lrp"
        nt_samples: If None, return raw attr. If >0, apply NoiseTunnel.
        nt_stdevs: Noise standard deviation
    
    Returns:
        Attribution map (1, 1, 28, 28), normalized to [0,1]
    """
    model.eval()
    
    # Compute target class first (without gradients)
    with torch.no_grad():
        logits = model(image)
        target_class = int(logits.argmax(dim=1).item())
    
    # Now enable gradients for attribution
    image.requires_grad_(True)
    
    # Select explainer
    if method == "ig":
        explainer = IntegratedGradients(model)
    elif method == "deeplift":
        explainer = DeepLift(model)
    elif method == "lrp":
        explainer = LRP(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute attribution
    try:
        if nt_samples is None or nt_samples <= 1:
            # Raw attribution
            attr = explainer.attribute(image, target=target_class)
        else:
            # Apply NoiseTunnel (SmoothGrad)
            nt = NoiseTunnel(explainer)
            attr = nt.attribute(image, nt_samples=nt_samples, stdevs=nt_stdevs, target=target_class)
    except Exception as e:
        print(f"  ⚠ Attribution failed: {str(e)[:100]}")
        attr = torch.zeros_like(image)
    
    # Normalize
    attr_abs = torch.abs(attr)
    attr_norm = attr_abs / (attr_abs.max() + 1e-8)
    
    return attr_norm.detach()

def compute_smoothness(attr):
    """
    Compute smoothness of attribution map using Total Variation.
    Lower TV = smoother.
    """
    attr_sq = attr.squeeze()
    # Compute gradients
    dy = torch.diff(attr_sq, dim=0)
    dx = torch.diff(attr_sq, dim=1)
    # Total variation
    tv = (torch.abs(dy).sum() + torch.abs(dx).sum()).item()
    return tv

def compute_sensitivity_to_attribution_removal(model, image, attr, num_masks=10, top_k_frac=0.2):
    """
    Compute sensitivity as model output variance when removing top attribution regions.
    High variance = sensitive (bad), Low variance = robust (good).
    
    Args:
        model: Neural network
        image: Input image (1, 1, 28, 28)
        attr: Attribution map (1, 1, 28, 28)
        num_masks: Number of random masks to test
        top_k_frac: Fraction of top pixels to mask
    
    Returns:
        Variance across outputs when top attribution regions are masked
    """
    model.eval()
    with torch.no_grad():
        # Original prediction
        orig_logits = model(image).detach()
        orig_pred = orig_logits.argmax(dim=1).item()
        
        # Get top-k pixels by attribution
        attr_flat = attr.squeeze().flatten()
        top_k = max(1, int(len(attr_flat) * top_k_frac))
        top_indices = torch.topk(attr_flat, k=top_k)[1]
        
        # Compute model output variance when masking these regions
        masked_logits = []
        for _ in range(num_masks):
            masked_img = image.clone()
            # Mask top attribution pixels with random noise
            for idx in top_indices:
                masked_img.view(-1)[idx] = torch.randn(1, device=DEVICE) * 0.3
            
            logits = model(masked_img).detach()
            masked_logits.append(logits)
        
        # Stack and compute variance
        logits_stack = torch.stack(masked_logits, dim=0)  # (num_masks, batch, 10)
        variance = logits_stack.var(dim=0).mean().item()  # Average variance across classes
    
    return variance

# ============================================================================
# Main Tuning Loop
# ============================================================================
def main():
    print("=" * 80)
    print("SmoothGrad Parameter Tuning for MNIST")
    print("=" * 80)
    
    # Load model and data
    model = load_mnist_model()
    images, labels = load_mnist_data(n_samples=N_INSTANCES)
    print(f"\n✓ Loaded {N_INSTANCES} MNIST instances")
    
    # Results storage
    results = []
    
    print(f"\n📊 Tuning SmoothGrad parameters...")
    print(f"  nt_samples grid: {NT_SAMPLES_GRID}")
    print(f"  nt_stdevs grid: {NT_STDEVS_GRID}")
    print(f"  Total configs: {len(NT_SAMPLES_GRID) * len(NT_STDEVS_GRID)}\n")
    
    # Iterate over parameter combinations
    config_num = 0
    total_configs = len(NT_SAMPLES_GRID) * len(NT_STDEVS_GRID)
    
    for nt_samples in NT_SAMPLES_GRID:
        for nt_stdevs in NT_STDEVS_GRID:
            config_num += 1
            smoothness_scores = []
            sensitivity_scores = []
            
            # Compute for each instance
            for i, (image, label) in enumerate(zip(images, labels)):
                image_single = image.unsqueeze(0)
                
                # Compute raw and smooth attributions (using IG for now)
                try:
                    attr_raw = compute_attribution(model, image_single, method="ig", nt_samples=None)
                    attr_smooth = compute_attribution(model, image_single, method="ig", 
                                                     nt_samples=nt_samples, nt_stdevs=nt_stdevs)
                    
                    # Compute smoothness (Total Variation)
                    smoothness = compute_smoothness(attr_smooth)
                    smoothness_scores.append(smoothness)
                    
                    # Compute sensitivity (model variance when masking top-attribution regions)
                    sensitivity = compute_sensitivity_to_attribution_removal(model, image_single, attr_smooth)
                    sensitivity_scores.append(sensitivity)
                except Exception as e:
                    print(f"  ⚠ Error processing instance {i}: {str(e)[:60]}")
                    continue
            
            if len(smoothness_scores) == 0:
                continue
            
            # Average metrics across instances
            mean_smoothness = np.mean(smoothness_scores)
            mean_sensitivity = np.mean(sensitivity_scores)
            
            results.append({
                'nt_samples': nt_samples,
                'nt_stdevs': nt_stdevs,
                'mean_smoothness': mean_smoothness,  # Lower = smoother
                'std_smoothness': np.std(smoothness_scores),
                'mean_sensitivity': mean_sensitivity,  # Lower = more robust
                'std_sensitivity': np.std(sensitivity_scores),
            })
            
            print(f"  [{config_num:2d}/{total_configs}] nt_samples={nt_samples:2d}, nt_stdevs={nt_stdevs:.2f} | "
                  f"Smoothness={mean_smoothness:.4f}±{np.std(smoothness_scores):.4f}, "
                  f"Sensitivity={mean_sensitivity:.6f}±{np.std(sensitivity_scores):.6f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "param_tuning_results.csv", index=False)
    
    print(f"\n✓ Results saved to {OUTPUT_DIR}/param_tuning_results.csv")
    print("\n" + "=" * 80)
    print("🏆 Top 5 Configurations (by Smoothness - lower is smoother)")
    print("=" * 80)
    top_smooth = results_df.nsmallest(5, 'mean_smoothness')
    print(top_smooth.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("� Top 5 Configurations (by Sensitivity - lower is more robust)")
    print("=" * 80)
    best_sensitivity = results_df.nsmallest(5, 'mean_sensitivity')
    print(best_sensitivity.to_string(index=False))
    
    # Visualization
    print("\n" + "=" * 80)
    print("📈 Generating visualizations...")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap: Smoothness vs params
    grid_smooth = results_df.pivot_table(values='mean_smoothness', 
                                        index='nt_samples', columns='nt_stdevs')
    sns.heatmap(grid_smooth, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0], 
                cbar_kws={'label': 'Total Variation (lower = smoother)'})
    axes[0].set_title('Attribution Smoothness\n(Lower = Smoother)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Noise Strength (nt_stdevs)')
    axes[0].set_ylabel('Num Samples (nt_samples)')
    
    # Heatmap: Sensitivity vs params
    grid_sens = results_df.pivot_table(values='mean_sensitivity', 
                                       index='nt_samples', columns='nt_stdevs')
    sns.heatmap(grid_sens, annot=True, fmt='.6f', cmap='RdYlGn_r', ax=axes[1], 
                cbar_kws={'label': 'Model Output Variance (lower = robust)'})
    axes[1].set_title('Model Sensitivity to Attribution Masking\n(Lower = More Robust)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Noise Strength (nt_stdevs)')
    axes[1].set_ylabel('Num Samples (nt_samples)')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "param_tuning_heatmaps.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved heatmaps to {OUTPUT_DIR}/param_tuning_heatmaps.png")
    plt.close()
    
    # Scatter plot: Smoothness vs Sensitivity trade-off
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(results_df['mean_smoothness'], results_df['mean_sensitivity'], 
                        c=results_df['nt_stdevs'], s=200, alpha=0.6, cmap='viridis', 
                        edgecolors='black', linewidth=1.5)
    
    # Annotate points
    for idx, row in results_df.iterrows():
        ax.annotate(f"({int(row['nt_samples'])}, {row['nt_stdevs']:.2f})", 
                   xy=(row['mean_smoothness'], row['mean_sensitivity']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Mean Smoothness (Lower = Smoother)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Sensitivity (Lower = More Robust)', fontsize=11, fontweight='bold')
    ax.set_title('SmoothGrad Parameter Trade-off: Smoothness vs Robustness', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower smoothness on right
    cbar = plt.colorbar(scatter, ax=ax, label='Noise Strength (nt_stdevs)')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "param_tuning_tradeoff.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved trade-off plot to {OUTPUT_DIR}/param_tuning_tradeoff.png")
    plt.close()
    
    # ========================================================================
    # Visual Inspection: Generate side-by-side comparisons for selected configs
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎨 Generating visual comparison plots...")
    print("=" * 80)
    
    # Select a few representative instances for visualization
    viz_indices = list(range(min(3, N_INSTANCES)))  # First few instances
    
    # Select best and worst configs + some in between
    selected_configs = [
        (results_df.nsmallest(1, 'mean_smoothness').iloc[0][['nt_samples', 'nt_stdevs']].values),  # Smoothest
        (results_df.nsmallest(1, 'mean_sensitivity').iloc[0][['nt_samples', 'nt_stdevs']].values),  # Most robust
        (4, 0.05),    # Minimal smoothing (if exists)
        (8, 0.10),    # Baseline-like
    ]
    
    for config_idx, config in enumerate(selected_configs):
        nt_samples, nt_stdevs = int(config[0]), float(config[1])
        
        # Check if this config exists in results
        if not any((results_df['nt_samples'] == nt_samples) & (results_df['nt_stdevs'] == nt_stdevs)):
            continue
        
        rows_for_montage = []

        def to_rgb_panel(arr, mode="inferno", gamma=0.4):
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.power(arr, gamma)

            if mode == "gray":
                rgb = np.stack([arr, arr, arr], axis=-1)
            elif mode == "gray_image":
                # Keep digit polarity intuitive for normalized MNIST input
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max - arr_min > 1e-8:
                    arr = (arr - arr_min) / (arr_max - arr_min)
                rgb = np.stack([arr, arr, arr], axis=-1)
            elif mode == "diff":
                rgb = plt.cm.YlOrRd(arr)[..., :3]
            else:
                rgb = plt.cm.inferno(arr)[..., :3]

            rgb_u8 = (rgb * 255).astype(np.uint8)
            return np.array(Image.fromarray(rgb_u8).resize((280, 280), resample=Image.Resampling.NEAREST))
        
        for row, img_idx in enumerate(viz_indices):
            image = images[img_idx].unsqueeze(0)
            
            try:
                attr_raw = compute_attribution(model, image, method="ig", nt_samples=None)
                attr_smooth = compute_attribution(model, image, method="ig", 
                                                nt_samples=nt_samples, nt_stdevs=nt_stdevs)
                
                tv_raw = compute_smoothness(attr_raw)
                tv_smooth = compute_smoothness(attr_smooth)
                tv_delta = ((tv_smooth - tv_raw) / tv_raw * 100) if tv_raw > 0 else 0

                input_np = image.squeeze(0).squeeze(0).detach().cpu().numpy()
                input_min, input_max = input_np.min(), input_np.max()
                if input_max - input_min > 1e-8:
                    input_np = (input_np - input_min) / (input_max - input_min)
                else:
                    input_np = np.zeros_like(input_np)

                raw_np = attr_raw.squeeze(0).squeeze(0).detach().cpu().numpy()
                smooth_np = attr_smooth.squeeze(0).squeeze(0).detach().cpu().numpy()
                diff_np = np.abs(smooth_np - raw_np)

                input_panel = to_rgb_panel(input_np, mode="gray_image", gamma=1.0)
                raw_panel = to_rgb_panel(raw_np, mode="inferno", gamma=0.4)
                smooth_panel = to_rgb_panel(smooth_np, mode="inferno", gamma=0.4)
                diff_panel = to_rgb_panel(diff_np, mode="diff", gamma=0.4)

                row_strip = np.concatenate([input_panel, raw_panel, smooth_panel, diff_panel], axis=1)
                rows_for_montage.append(row_strip)

                print(
                    f"    Instance {img_idx}: TV raw={tv_raw:.2f}, smooth={tv_smooth:.2f}, Δ={tv_delta:+.1f}%"
                )
            except Exception as e:
                print(f"    ⚠ Instance {img_idx} visualization failed: {str(e)[:80]}")

        if len(rows_for_montage) == 0:
            continue

        montage = np.concatenate(rows_for_montage, axis=0)
        montage_img = Image.fromarray(montage)
        montage_img.save(OUTPUT_DIR / f"visual_comparison_config_{config_idx}.png")
        print(f"✓ Saved visual comparison {config_idx}: nt_samples={nt_samples}, nt_stdevs={nt_stdevs}")
    
    print("\n" + "=" * 80)
    print("✅ Parameter tuning complete!")
    print(f"📁 Results and visualizations saved to {OUTPUT_DIR}")
    print("=" * 80)
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("💡 Recommendations:")
    print("=" * 80)
    smoothest = results_df.nsmallest(1, 'mean_smoothness').iloc[0]
    most_robust = results_df.nsmallest(1, 'mean_sensitivity').iloc[0]
    
    print(f"\n🎯 For maximum smoothness:")
    print(f"   nt_samples={int(smoothest['nt_samples'])}, nt_stdevs={smoothest['nt_stdevs']:.2f}")
    print(f"   → Smoothness: {smoothest['mean_smoothness']:.4f}, Sensitivity: {smoothest['mean_sensitivity']:.6f}")
    
    print(f"\n💪 For maximum robustness (minimal sensitivity):")
    print(f"   nt_samples={int(most_robust['nt_samples'])}, nt_stdevs={most_robust['nt_stdevs']:.2f}")
    print(f"   → Smoothness: {most_robust['mean_smoothness']:.4f}, Sensitivity: {most_robust['mean_sensitivity']:.6f}")
    
    # Find Pareto frontier (good in both)
    results_df['score'] = (
        (1 - (results_df['mean_smoothness'] - results_df['mean_smoothness'].min()) / 
              (results_df['mean_smoothness'].max() - results_df['mean_smoothness'].min() + 1e-8)) +
        (1 - (results_df['mean_sensitivity'] - results_df['mean_sensitivity'].min()) / 
         (results_df['mean_sensitivity'].max() - results_df['mean_sensitivity'].min() + 1e-8))
    ) / 2
    
    best_balanced = results_df.nlargest(1, 'score').iloc[0]
    print(f"\n⚖️ Best balanced (smoothness + robustness):")
    print(f"   nt_samples={int(best_balanced['nt_samples'])}, nt_stdevs={best_balanced['nt_stdevs']:.2f}")
    print(f"   → Smoothness: {best_balanced['mean_smoothness']:.4f}, Sensitivity: {best_balanced['mean_sensitivity']:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
