#!/usr/bin/env python3
"""
Generate MNIST explanations using the same three explainers used elsewhere:
- Integrated Gradients (ig)
- DeepLift (dl)
- LRP (lrp)

Outputs a single artifact with:
- selected test images
- true labels
- predicted labels + confidence
- attribution maps for each explainer

Usage:
    python mnist_explain_predictions.py \
        --model-path models/mnist/mnist_best_model.pt \
        --n-samples 128
"""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, IntegratedGradients
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from explainers.lrp import LRP
from train_mnist_model import MNISTNet, set_seed


def _configure_ssl_for_macos() -> None:
    """Allow torchvision MNIST download in environments with broken cert chains."""
    ssl._create_default_https_context = ssl._create_unverified_context


def load_mnist_model(model_path: Path, device: torch.device) -> MNISTNet:
    """Load trained MNIST model checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = MNISTNet()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.to(device)
    model.eval()
    return model


def load_test_subset(data_dir: Path, n_samples: int, seed: int) -> Tuple[DataLoader, np.ndarray]:
    """Load deterministic subset of MNIST test split."""
    _configure_ssl_for_macos()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    n_total = len(test_dataset)
    n_use = min(n_samples, n_total)

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_total, size=n_use, replace=False)

    subset = Subset(test_dataset, sample_indices.tolist())
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

    return loader, sample_indices


def compute_baseline(data_dir: Path, baseline_type: str, device: torch.device) -> torch.Tensor:
    """Compute baseline for IG/DL explainers."""
    if baseline_type == "zero":
        return torch.zeros(1, 1, 28, 28, dtype=torch.float32, device=device)

    if baseline_type != "mean":
        raise ValueError(f"Unsupported baseline_type: {baseline_type}")

    _configure_ssl_for_macos()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)

    pixel_sum = 0.0
    count = 0
    for images, _ in loader:
        pixel_sum += images.sum().item()
        count += images.numel()

    mean_pixel = pixel_sum / max(count, 1)
    baseline = torch.full((1, 1, 28, 28), fill_value=float(mean_pixel), dtype=torch.float32, device=device)
    return baseline


class MNISTLogitsWrapper(torch.nn.Module):
    """Expose logits from MNISTNet for attribution methods that need pre-softmax outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_logits=True)


def build_explainers(model: torch.nn.Module, methods: List[str]) -> Dict[str, object]:
    """Instantiate requested explainers."""
    explainers: Dict[str, object] = {}
    logits_model = MNISTLogitsWrapper(model)

    for method in methods:
        if method == "ig":
            explainers[method] = IntegratedGradients(model)
        elif method == "dl":
            explainers[method] = DeepLift(logits_model)
        elif method == "lrp":
            explainers[method] = LRP(logits_model)
        else:
            raise ValueError(f"Unknown explainer method: {method}")

    return explainers


def explain_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    explainers: Dict[str, object],
    baseline: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run prediction + attribution generation over the subset."""
    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_conf: List[torch.Tensor] = []
    attr_store: Dict[str, List[torch.Tensor]] = {method: [] for method in explainers}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

        for method, explainer in explainers.items():
            xb = images.clone().detach().requires_grad_(True)

            if method in {"ig", "dl"}:
                b = baseline.expand_as(xb)
                attrs = explainer.attribute(xb, baselines=b, target=preds)
            else:
                attrs = explainer.attribute(xb, target=preds)

            attr_store[method].append(attrs.detach().cpu())

        all_images.append(images.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_conf.append(conf.detach().cpu())

    output = {
        "images": torch.cat(all_images, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "pred_labels": torch.cat(all_preds, dim=0),
        "confidences": torch.cat(all_conf, dim=0),
        "attributions": {m: torch.cat(chunks, dim=0) for m, chunks in attr_store.items()},
    }
    return output


def save_artifacts(
    payload: Dict[str, torch.Tensor],
    sample_indices: np.ndarray,
    methods: List[str],
    baseline: torch.Tensor,
    baseline_type: str,
    model_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save explanation artifact and prediction manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / "mnist_explanations.pt"
    manifest_path = output_dir / "mnist_predictions.csv"

    artifact = {
        "images": payload["images"],
        "labels": payload["labels"],
        "pred_labels": payload["pred_labels"],
        "confidences": payload["confidences"],
        "attributions": payload["attributions"],
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "methods": methods,
        "baseline": baseline.detach().cpu(),
        "baseline_type": baseline_type,
        "model_path": str(model_path),
    }

    torch.save(artifact, artifact_path)

    manifest_df = pd.DataFrame(
        {
            "sample_position": np.arange(len(sample_indices)),
            "original_test_index": sample_indices,
            "true_label": payload["labels"].numpy().astype(int),
            "pred_label": payload["pred_labels"].numpy().astype(int),
            "confidence": payload["confidences"].numpy().astype(float),
        }
    )
    manifest_df.to_csv(manifest_path, index=False)

    config_path = output_dir / "mnist_explanation_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "methods": methods,
                "baseline_type": baseline_type,
                "n_samples": int(len(sample_indices)),
                "model_path": str(model_path),
                "artifact_path": str(artifact_path),
                "manifest_path": str(manifest_path),
            },
            f,
            indent=2,
        )

    return artifact_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MNIST explanations (ig, dl, lrp).")
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist/mnist_best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("explanations/mnist"))
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-type", choices=["zero", "mean"], default="zero")
    parser.add_argument("--methods", nargs="+", default=["ig", "dl", "lrp"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    methods = [m.lower() for m in args.methods]
    for method in methods:
        if method not in {"ig", "dl", "lrp"}:
            raise ValueError(f"Unsupported method: {method}")

    print("Loading model...")
    model = load_mnist_model(args.model_path, device)

    print("Loading MNIST subset...")
    loader, sample_indices = load_test_subset(args.data_dir, args.n_samples, args.seed)

    print(f"Computing baseline ({args.baseline_type})...")
    baseline = compute_baseline(args.data_dir, args.baseline_type, device)

    print(f"Building explainers: {methods}")
    explainers = build_explainers(model, methods)

    print("Generating attributions...")
    payload = explain_subset(model, loader, explainers, baseline, device)

    print("Saving artifacts...")
    artifact_path, manifest_path = save_artifacts(
        payload=payload,
        sample_indices=sample_indices,
        methods=methods,
        baseline=baseline,
        baseline_type=args.baseline_type,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    print("✅ Explanation generation complete")
    print(f"Artifact: {artifact_path}")
    print(f"Predictions CSV: {manifest_path}")


if __name__ == "__main__":
    main()
