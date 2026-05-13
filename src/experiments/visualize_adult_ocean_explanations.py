#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from explainers.captum_explain import explain_batch
from load.load_net import load_net


def load_model(dataset: str, model_name: str = "model1"):
    architecture_by_model = {
        "model1": "smallNN",
        "model2": "deeperNN",
        "model3": "shallowNN",
    }
    net_module = load_net(dataset)
    model = net_module.recover_net(architecture_by_model[model_name])
    model_path = ROOT / "models" / dataset / f"{dataset}_{model_name}.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def load_data(dataset: str):
    return torch.load(ROOT / "src" / "data" / dataset / "Xtest.pt", map_location="cpu").float()


def sample_tensor(X: torch.Tensor, max_samples: int, seed: int = 42):
    if X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx]


def main():
    datasets = ["adult", "ocean"]
    methods = ["ig", "dl"]
    model_name = "model1"
    max_samples = 1000

    out_dir = ROOT / "src" / "results" / "explanation_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs = {}
    for dataset in datasets:
        X = load_data(dataset)
        Xs = sample_tensor(X, max_samples=max_samples, seed=42)
        model = load_model(dataset, model_name=model_name)
        attrs[dataset] = explain_batch(model, Xs, methods=methods, baselines=None, batch_size=256)
        print(f"Loaded {dataset}: X={tuple(Xs.shape)}")

    # 1) Distribution of log10(|attr| + eps)
    eps = 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, method in enumerate(methods):
        ax = axes[col]
        for dataset in datasets:
            arr = attrs[dataset][method].detach().cpu().numpy()
            log_abs = np.log10(np.abs(arr).reshape(-1) + eps)
            ax.hist(log_abs, bins=120, alpha=0.45, density=True, label=dataset)
        ax.set_title(f"{method.upper()} attribution magnitude")
        ax.set_xlabel("log10(|attribution| + 1e-12)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Adult vs Ocean explanation magnitude distributions (model1)", y=1.02)
    fig.tight_layout()
    p1 = out_dir / "adult_vs_ocean_log_abs_distribution.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) Near-zero curve
    thresholds = np.logspace(-10, -1, 80)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, method in enumerate(methods):
        ax = axes[col]
        for dataset in datasets:
            arr = attrs[dataset][method].detach().cpu().numpy()
            abs_arr = np.abs(arr)
            frac = [float((abs_arr <= t).mean()) for t in thresholds]
            ax.plot(thresholds, frac, label=dataset, linewidth=2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(f"{method.upper()} near-zero fraction curve")
        ax.set_xlabel("Threshold t (fraction of |attr| <= t)")
        ax.set_ylabel("Fraction near-zero")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Near-zero attribution rate comparison (model1)", y=1.02)
    fig.tight_layout()
    p2 = out_dir / "adult_vs_ocean_near_zero_curves.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) Heatmaps of raw attributions
    n_rows = 120
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            arr = attrs[dataset][method].detach().cpu().numpy()
            block = arr[:n_rows]
            vmax = np.percentile(np.abs(block), 99)
            if vmax == 0:
                vmax = 1e-8
            im = ax.imshow(block, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{dataset} - {method.upper()}")
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Sample index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Attribution heatmaps (first 120 samples, model1)", y=1.02)
    fig.tight_layout()
    p3 = out_dir / "adult_vs_ocean_attribution_heatmaps.png"
    fig.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
