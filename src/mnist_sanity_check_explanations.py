#!/usr/bin/env python3
"""
Generate MNIST sanity-check explanations via incremental layer randomization.

By default, only the last parameterized layer is randomized (the output layer),
which is the current focus for sanity checking.

Outputs:
- explanations/mnist/mnist_sanity_explanations.pt
- explanations/mnist/mnist_sanity_check_config.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from captum.attr import DeepLift, IntegratedGradients

from explainers.lrp import LRP
from train_mnist_model import MNISTNet, set_seed


class MNISTLogitsWrapper(torch.nn.Module):
    """Expose logits from MNISTNet for attribution methods that need pre-softmax outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_logits=True)


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


def build_explainers(model: torch.nn.Module, methods: List[str]) -> Dict[str, object]:
    """Instantiate explainers with logits wrappers for DeepLift and LRP."""
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


def get_parameterized_layer_names(model: torch.nn.Module) -> List[str]:
    """Return names of leaf modules with direct parameters in model traversal order."""
    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        params = list(module.parameters(recurse=False))
        if params:
            names.append(name)
    if not names:
        raise RuntimeError("No parameterized layers found in model.")
    return names


def stable_layer_seed(base_seed: int, layer_name: str) -> int:
    """Deterministic per-layer seed independent of Python hash randomization."""
    offset = int(np.frombuffer(layer_name.encode("utf-8"), dtype=np.uint8).sum())
    return int((base_seed + offset) % (2**31 - 1))


def randomize_module_parameters(module: torch.nn.Module, seed: int) -> None:
    """Randomize module parameters, preferring native reset_parameters when available."""
    cpu_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    else:
        for param in module.parameters(recurse=False):
            if param.ndim > 1:
                torch.nn.init.kaiming_uniform_(param, a=np.sqrt(5))
            else:
                bound = 1.0 / max(np.sqrt(param.numel()), 1.0)
                torch.nn.init.uniform_(param, -bound, bound)

    torch.random.set_rng_state(cpu_state)


def randomize_layers(model: torch.nn.Module, layer_names: List[str], base_seed: int) -> None:
    """Randomize listed layers in-place on model."""
    module_map = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in module_map:
            raise KeyError(f"Layer {layer_name} not found in model.")
        randomize_module_parameters(module_map[layer_name], stable_layer_seed(base_seed, layer_name))


def compute_attributions(
    model: torch.nn.Module,
    methods: List[str],
    images: torch.Tensor,
    targets: torch.Tensor,
    baseline: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute attributions for all requested methods on fixed target classes."""
    explainers = build_explainers(model, methods)
    attrs: Dict[str, torch.Tensor] = {}

    x = images.to(device)
    t = targets.to(device)

    for method, explainer in explainers.items():
        xb = x.clone().detach().requires_grad_(True)

        if method in {"ig", "dl"}:
            b = baseline.to(device).expand_as(xb)
            out = explainer.attribute(xb, baselines=b, target=t)
        else:
            out = explainer.attribute(xb, target=t)

        attrs[method] = out.detach().cpu()

    return attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MNIST sanity-check explanations.")
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=Path("explanations/mnist/mnist_explanations.pt"),
        help="Base explanation artifact to reuse image subset and targets.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/mnist/mnist_best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("explanations/mnist"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="mnist_sanity_explanations.pt",
    )
    parser.add_argument(
        "--layer-scope",
        choices=["last", "all"],
        default="last",
        help="Randomization scope. 'all' randomizes from output to input incrementally.",
    )
    parser.add_argument(
        "--layer-names",
        nargs="+",
        default=None,
        help="Optional explicit layer randomization order (output to input preferred).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to explain (default: methods from source artifact).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["artifact_pred", "randomized_pred"],
        default="artifact_pred",
        help="Whether to explain original predicted class or each randomized model's predicted class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.source_artifact.exists():
        raise FileNotFoundError(f"Source artifact not found: {args.source_artifact}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_artifact = torch.load(args.source_artifact, map_location="cpu")

    required_keys = ["images", "labels", "pred_labels", "confidences", "sample_indices", "baseline"]
    missing = [key for key in required_keys if key not in source_artifact]
    if missing:
        raise RuntimeError(f"Source artifact is missing required keys: {missing}")

    methods = [m.lower() for m in (args.methods or source_artifact.get("methods", ["ig", "dl", "lrp"]))]
    for method in methods:
        if method not in {"ig", "dl", "lrp"}:
            raise ValueError(f"Unsupported method: {method}")

    base_model = load_mnist_model(args.model_path, device)
    available_layers = get_parameterized_layer_names(base_model)

    if args.layer_names:
        randomization_order = args.layer_names
    elif args.layer_scope == "last":
        randomization_order = [available_layers[-1]]
    else:
        randomization_order = list(reversed(available_layers))

    invalid = [name for name in randomization_order if name not in available_layers]
    if invalid:
        raise ValueError(
            f"Unknown layer names in randomization order: {invalid}. Available: {available_layers}"
        )

    print(f"Available parameterized layers: {available_layers}")
    print(f"Randomization order (incremental): {randomization_order}")

    images = source_artifact["images"].float()
    labels = source_artifact["labels"].long()
    artifact_pred_labels = source_artifact["pred_labels"].long()
    confidences = source_artifact["confidences"].float()
    sample_indices = source_artifact["sample_indices"].long()
    baseline = source_artifact["baseline"].float()

    steps = []

    for step_idx in range(len(randomization_order)):
        randomized_layers = randomization_order[: step_idx + 1]
        randomized_model = copy.deepcopy(base_model).to(device)
        randomized_model.eval()
        randomize_layers(randomized_model, randomized_layers, args.seed)

        with torch.no_grad():
            logits = randomized_model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            rand_conf, rand_pred = probs.max(dim=1)

        if args.target_mode == "randomized_pred":
            targets = rand_pred.detach().cpu().long()
        else:
            targets = artifact_pred_labels.clone()

        attrs = compute_attributions(
            model=randomized_model,
            methods=methods,
            images=images,
            targets=targets,
            baseline=baseline,
            device=device,
        )

        step_name = "+".join(randomized_layers)
        print(f"Computed sanity step {step_idx + 1}/{len(randomization_order)}: {step_name}")

        steps.append(
            {
                "step_index": step_idx,
                "step_name": step_name,
                "randomized_layers": randomized_layers,
                "target_mode": args.target_mode,
                "target_labels": targets,
                "pred_labels": rand_pred.detach().cpu(),
                "confidences": rand_conf.detach().cpu(),
                "attributions": attrs,
            }
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / args.output_name
    config_path = out_dir / "mnist_sanity_check_config.json"

    sanity_artifact = {
        "source_artifact_path": str(args.source_artifact),
        "model_path": str(args.model_path),
        "methods": methods,
        "images": images,
        "labels": labels,
        "pred_labels": artifact_pred_labels,
        "confidences": confidences,
        "sample_indices": sample_indices,
        "baseline": baseline,
        "baseline_type": source_artifact.get("baseline_type", "zero"),
        "randomization_order": randomization_order,
        "available_layers": available_layers,
        "steps": steps,
    }

    torch.save(sanity_artifact, artifact_path)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_artifact_path": str(args.source_artifact),
                "model_path": str(args.model_path),
                "methods": methods,
                "n_samples": int(images.shape[0]),
                "layer_scope": args.layer_scope,
                "randomization_order": randomization_order,
                "target_mode": args.target_mode,
                "artifact_path": str(artifact_path),
            },
            f,
            indent=2,
        )

    print("✅ Sanity-check explanation generation complete")
    print(f"Sanity artifact: {artifact_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
