#!/usr/bin/env python3
"""
Evaluate MNIST explanations with:
- Case Align (S_plus, R_bounded) using full-test retrieval
- Captum sensitivity (sensitivity_max)

Expected input artifact from:
    python mnist_explain_predictions.py

Outputs:
- explanations/mnist/mnist_explanation_scores.csv
- explanations/mnist/mnist_method_summary.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import ssl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, IntegratedGradients
from captum.metrics import sensitivity_max
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from case_align.metrics import rankdata, safe_normalise_rows
from explainers.lrp import LRP
from train_mnist_model import MNISTNet, set_seed


def load_model(model_path: Path, device: torch.device) -> MNISTNet:
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
        raise RuntimeError("Unsupported checkpoint format")

    model.to(device)
    model.eval()
    return model


def _configure_ssl_for_macos() -> None:
    """Allow torchvision MNIST download in environments with broken cert chains."""
    ssl._create_default_https_context = ssl._create_unverified_context


def load_full_test_split(data_dir: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Load the full MNIST test split used as retrieval pool."""
    _configure_ssl_for_macos()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)

    images_tensor = torch.cat(all_images, dim=0).float()
    labels_np = torch.cat(all_labels, dim=0).numpy().astype(int)
    return images_tensor, labels_np


def predict_labels_for_images(
    model: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict labels/confidences for a tensor of images."""
    pred_chunks: List[np.ndarray] = []
    conf_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            xb = images[start:end].to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            pred_chunks.append(preds.detach().cpu().numpy().astype(int))
            conf_chunks.append(conf.detach().cpu().numpy().astype(float))

    pred_labels = np.concatenate(pred_chunks, axis=0)
    confidences = np.concatenate(conf_chunks, axis=0)
    return pred_labels, confidences


def compute_retrieval_attributions(
    explainer: object,
    method: str,
    images: torch.Tensor,
    pred_labels: np.ndarray,
    baseline: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate attributions for retrieval pool in batches (not persisted to disk)."""
    n_total = len(images)
    n_batches = (n_total + batch_size - 1) // batch_size
    chunks: List[torch.Tensor] = []

    for batch_idx, start in enumerate(range(0, n_total, batch_size), start=1):
        end = min(start + batch_size, n_total)
        xb = images[start:end].to(device).clone().detach().requires_grad_(True)
        targets = torch.from_numpy(pred_labels[start:end]).to(device)

        if method in {"ig", "dl"}:
            b = baseline.to(device).expand_as(xb)
            attrs = explainer.attribute(xb, baselines=b, target=targets)
        else:
            attrs = explainer.attribute(xb, target=targets)

        chunks.append(attrs.detach().cpu())

        if batch_idx == 1 or batch_idx == n_batches or batch_idx % max(n_batches // 10, 1) == 0:
            print(f"    retrieval attribution batches: {batch_idx}/{n_batches}")

    return torch.cat(chunks, dim=0).float()


def build_metric_context(matrix: np.ndarray, sim_metric: str, epsilon: float = 1e-8) -> Dict[str, np.ndarray]:
    """Prepare reusable structures for row-wise distance calculations."""
    matrix_raw = np.asarray(matrix, dtype=float)
    context: Dict[str, np.ndarray] = {
        "matrix_raw": matrix_raw,
        "epsilon": np.array([epsilon], dtype=float),
    }

    if sim_metric == "gower":
        ranges = np.ptp(matrix_raw, axis=0)
        context["ranges"] = np.where(ranges == 0, 1.0, ranges)
    elif sim_metric == "cosine":
        context["matrix_repr"] = safe_normalise_rows(matrix_raw, eps=epsilon)
    elif sim_metric == "spearman":
        ranked = np.vstack([rankdata(row) for row in matrix_raw])
        ranked = ranked - ranked.mean(axis=1, keepdims=True)
        context["matrix_repr"] = safe_normalise_rows(ranked, eps=epsilon)
    else:
        raise ValueError(f"Unknown sim_metric: {sim_metric}")

    context["sim_metric"] = np.array([0], dtype=float)
    context["sim_metric_name"] = np.array([sim_metric], dtype=object)
    return context


def row_distances(query_vec: np.ndarray, context: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute distances from query vector to all rows in context matrix."""
    sim_metric = str(context["sim_metric_name"][0])
    epsilon = float(context["epsilon"][0])
    matrix_raw = context["matrix_raw"]

    if sim_metric == "gower":
        ranges = context["ranges"]
        dist = np.mean(np.abs(matrix_raw - query_vec[None, :]) / ranges[None, :], axis=1)
        return np.clip(dist, 0.0, 1.0)

    if sim_metric == "cosine":
        matrix_repr = context["matrix_repr"]
        query_repr = safe_normalise_rows(np.asarray(query_vec, dtype=float)[None, :], eps=epsilon)[0]
        similarity = matrix_repr @ query_repr
        return np.clip(1.0 - 0.5 * (similarity + 1.0), 0.0, 1.0)

    if sim_metric == "spearman":
        matrix_repr = context["matrix_repr"]
        q_rank = rankdata(np.asarray(query_vec, dtype=float))
        q_rank = q_rank - q_rank.mean()
        query_repr = safe_normalise_rows(q_rank[None, :], eps=epsilon)[0]
        similarity = matrix_repr @ query_repr
        return np.clip(1.0 - 0.5 * (similarity + 1.0), 0.0, 1.0)

    raise ValueError(f"Unknown sim_metric: {sim_metric}")


def compute_case_align_like_only(
    query_index: int,
    query_label: int,
    retrieval_labels: np.ndarray,
    problem_context: Dict[str, np.ndarray],
    solution_context: Dict[str, np.ndarray],
    k: int,
    epsilon: float = 1e-8,
) -> Tuple[float, int, bool]:
    """Compute like-only Case Align S+ for one query index over full retrieval pool."""
    query_problem = problem_context["matrix_raw"][query_index]
    dprob_all = row_distances(query_problem, problem_context)

    like_mask = retrieval_labels == query_label
    like_mask[query_index] = False
    like_indices = np.where(like_mask)[0]
    if like_indices.size == 0:
        return 0.0, 0, False

    like_dists = dprob_all[like_indices]
    order = np.argsort(like_dists)
    k_use = min(k, like_indices.size)
    neigh_indices = like_indices[order[:k_use]]
    dprob_neigh = dprob_all[neigh_indices]

    query_solution = solution_context["matrix_raw"][query_index]
    dsoln_all = row_distances(query_solution, solution_context)
    dsoln_neigh = dsoln_all[neigh_indices]

    ds_min = float(np.min(dsoln_all))
    ds_max = float(np.max(dsoln_all))
    denom = max(ds_max - ds_min, epsilon)
    align = 1.0 - (dsoln_neigh - ds_min) / denom

    weights = 1.0 - dprob_neigh
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0, int(k_use), True

    s_plus = float(np.sum(weights * align) / weight_sum)
    return s_plus, int(k_use), True


class MNISTLogitsWrapper(torch.nn.Module):
    """Expose logits from MNISTNet for attribution methods that need pre-softmax outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_logits=True)


def build_explainers(model: torch.nn.Module, methods: List[str]) -> Dict[str, object]:
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
            raise ValueError(f"Unknown method: {method}")

    return explainers


def compute_sensitivity(
    explainer: object,
    method: str,
    image: torch.Tensor,
    target: int,
    baseline: torch.Tensor,
    perturb_radius: float,
    n_perturb_samples: int,
    device: torch.device,
) -> float:
    """Compute Captum sensitivity_max for one sample (strict: no fallback)."""
    x = image.unsqueeze(0).to(device)

    if method in {"ig", "dl"}:
        b0 = baseline.to(device)

        def explain_func(inputs, target=None):
            x_in = inputs[0] if isinstance(inputs, tuple) else inputs
            b = b0.expand_as(x_in)
            return explainer.attribute(x_in, baselines=b, target=target)

        sens = sensitivity_max(
            explanation_func=explain_func,
            inputs=x,
            target=target,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples,
        )
    else:
        sens = sensitivity_max(
            explanation_func=explainer.attribute,
            inputs=x,
            target=target,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples,
        )

    return float(sens.detach().cpu().item())


def evaluate_method(
    method: str,
    explainer: object,
    query_images: torch.Tensor,
    query_labels: np.ndarray,
    query_pred_labels: np.ndarray,
    query_confidences: np.ndarray,
    query_retrieval_indices: np.ndarray,
    retrieval_images: torch.Tensor,
    retrieval_labels: np.ndarray,
    retrieval_pred_labels: np.ndarray,
    baseline: torch.Tensor,
    k: int,
    sim_metric: str,
    perturb_radius: float,
    n_perturb_samples: int,
    retrieval_batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate one explainer with full-test retrieval and on-the-fly attribution generation."""
    n_queries = query_images.shape[0]
    n_retrieval = retrieval_images.shape[0]

    print(f"  Building retrieval attributions for {method} over {n_retrieval} samples...")
    retrieval_attributions = compute_retrieval_attributions(
        explainer=explainer,
        method=method,
        images=retrieval_images,
        pred_labels=retrieval_pred_labels,
        baseline=baseline,
        batch_size=retrieval_batch_size,
        device=device,
    )

    X_flat = retrieval_images.view(n_retrieval, -1).detach().cpu().numpy()
    E_flat = retrieval_attributions.view(n_retrieval, -1).detach().cpu().numpy()
    problem_context = build_metric_context(X_flat, sim_metric=sim_metric)
    solution_context = build_metric_context(E_flat, sim_metric=sim_metric)

    rows = []
    for i in range(n_queries):
        retrieval_index = int(query_retrieval_indices[i])
        if retrieval_index < 0 or retrieval_index >= n_retrieval:
            raise IndexError(
                f"Query retrieval index out of bounds: {retrieval_index} for pool size {n_retrieval}"
            )

        s_plus, like_count, has_like_neighbour = compute_case_align_like_only(
            query_index=retrieval_index,
            query_label=int(query_labels[i]),
            retrieval_labels=retrieval_labels,
            problem_context=problem_context,
            solution_context=solution_context,
            k=k,
        )

        sensitivity = compute_sensitivity(
            explainer=explainer,
            method=method,
            image=query_images[i],
            target=int(query_pred_labels[i]),
            baseline=baseline,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples,
            device=device,
        )

        rows.append(
            {
                "method": method,
                "sample_position": i,
                "original_test_index": retrieval_index,
                "true_label": int(query_labels[i]),
                "pred_label": int(query_pred_labels[i]),
                "confidence": float(query_confidences[i]),
                "case_align_like_count": int(like_count),
                "case_align_has_like_neighbour": bool(has_like_neighbour),
                "case_align_S_plus": float(s_plus) if has_like_neighbour else np.nan,
                "case_align_R_bounded": float(s_plus) if has_like_neighbour else np.nan,
                "captum_sensitivity": float(sensitivity),
            }
        )

    return pd.DataFrame(rows)


def summarise_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-method summary metrics and correlations."""
    summaries = []

    for method, group in df.groupby("method"):
        valid = group.dropna(subset=["case_align_S_plus", "captum_sensitivity"])

        if (
            len(valid) >= 2
            and valid["case_align_S_plus"].nunique() > 1
            and valid["captum_sensitivity"].nunique() > 1
        ):
            pearson_r, pearson_p = pearsonr(valid["case_align_S_plus"], valid["captum_sensitivity"])
            spearman_r, spearman_p = spearmanr(valid["case_align_S_plus"], valid["captum_sensitivity"])
        else:
            pearson_r = pearson_p = np.nan
            spearman_r = spearman_p = np.nan

        summaries.append(
            {
                "method": method,
                "n_samples": int(len(group)),
                "n_case_align_valid": int(len(valid)),
                "mean_case_align_S_plus": float(group["case_align_S_plus"].mean()),
                "mean_case_align_R_bounded": float(group["case_align_R_bounded"].mean()),
                "mean_captum_sensitivity": float(group["captum_sensitivity"].mean()),
                "std_captum_sensitivity": float(group["captum_sensitivity"].std()),
                "pearson_case_align_vs_sensitivity": float(pearson_r),
                "pearson_p_value": float(pearson_p),
                "spearman_case_align_vs_sensitivity": float(spearman_r),
                "spearman_p_value": float(spearman_p),
            }
        )

    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Helpers for reconstructing randomized models (mirrors mnist_sanity_check_explanations.py)
# ---------------------------------------------------------------------------

def _stable_layer_seed(base_seed: int, layer_name: str) -> int:
    """Deterministic per-layer seed independent of Python hash randomization."""
    offset = int(np.frombuffer(layer_name.encode("utf-8"), dtype=np.uint8).sum())
    return int((base_seed + offset) % (2**31 - 1))


def _randomize_module_parameters(module: torch.nn.Module, seed: int) -> None:
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


def _randomize_layers(model: torch.nn.Module, layer_names: List[str], base_seed: int) -> None:
    """Randomize listed layers in-place."""
    module_map = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in module_map:
            raise KeyError(f"Layer {layer_name!r} not found in model.")
        _randomize_module_parameters(module_map[layer_name], _stable_layer_seed(base_seed, layer_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MNIST explanation quality.")
    parser.add_argument("--artifact-path", type=Path, default=Path("explanations/mnist/mnist_explanations.pt"))
    parser.add_argument("--model-path", type=Path, default=Path("models/mnist/mnist_best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("explanations/mnist"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sim-metric", choices=["gower", "cosine", "spearman"], default="gower")
    parser.add_argument("--perturb-radius", type=float, default=0.1)
    parser.add_argument("--n-perturb-samples", type=int, default=10)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    # Sanity-check evaluation (optional)
    parser.add_argument(
        "--sanity-artifact",
        type=Path,
        default=None,
        help="Path to mnist_sanity_explanations.pt. When provided, also evaluates sanity step.",
    )
    parser.add_argument(
        "--sanity-step-index",
        type=int,
        default=-1,
        help="Index of the sanity step to evaluate (-1 = last step).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.artifact_path.exists():
        raise FileNotFoundError(
            f"Explanation artifact not found at {args.artifact_path}. "
            "Run mnist_explain_predictions.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    artifact = torch.load(args.artifact_path, map_location="cpu")
    methods = [m.lower() for m in artifact["methods"]]

    if "sample_indices" not in artifact:
        raise RuntimeError(
            "Artifact does not contain sample_indices. Regenerate with mnist_explain_predictions.py first."
        )

    query_images = artifact["images"].float()
    query_labels = artifact["labels"].numpy().astype(int)
    query_pred_labels = artifact["pred_labels"].numpy().astype(int)
    query_confidences = artifact["confidences"].numpy().astype(float)
    query_retrieval_indices = artifact["sample_indices"].numpy().astype(int)
    baseline = artifact["baseline"].float()

    model = load_model(args.model_path, device)
    explainers = build_explainers(model, methods)

    print("Loading full MNIST test split for retrieval...")
    retrieval_images, retrieval_labels = load_full_test_split(args.data_dir)
    retrieval_pred_labels, _ = predict_labels_for_images(
        model=model,
        images=retrieval_images,
        batch_size=args.retrieval_batch_size,
        device=device,
    )
    print(f"Retrieval pool size: {len(retrieval_labels)}")

    if np.any(query_retrieval_indices < 0) or np.any(query_retrieval_indices >= len(retrieval_labels)):
        raise IndexError("Some query sample_indices are outside the retrieval pool range.")

    retrieval_labels_for_queries = retrieval_labels[query_retrieval_indices]
    if not np.array_equal(retrieval_labels_for_queries, query_labels):
        print("⚠️ Query labels differ from retrieval labels at sample indices; using artifact labels for reporting.")

    all_scores = []

    for method in methods:
        print(f"Evaluating method: {method}")
        method_scores = evaluate_method(
            method=method,
            explainer=explainers[method],
            query_images=query_images,
            query_labels=query_labels,
            query_pred_labels=query_pred_labels,
            query_confidences=query_confidences,
            query_retrieval_indices=query_retrieval_indices,
            retrieval_images=retrieval_images,
            retrieval_labels=retrieval_labels,
            retrieval_pred_labels=retrieval_pred_labels,
            baseline=baseline,
            k=args.k,
            sim_metric=args.sim_metric,
            perturb_radius=args.perturb_radius,
            n_perturb_samples=args.n_perturb_samples,
            retrieval_batch_size=args.retrieval_batch_size,
            device=device,
        )
        all_scores.append(method_scores)

    scores_df = pd.concat(all_scores, ignore_index=True)
    summary_df = summarise_scores(scores_df)

    missing_case_align = int((~scores_df["case_align_has_like_neighbour"]).sum())
    if missing_case_align > 0:
        print(
            "ℹ️ Case Align unavailable for "
            f"{missing_case_align} row(s) due to missing same-class neighbours; saved as NaN."
        )

    scores_path = args.output_dir / "mnist_explanation_scores.csv"
    summary_path = args.output_dir / "mnist_method_summary.csv"
    report_path = args.output_dir / "mnist_evaluation_config.json"

    scores_df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_path": str(args.artifact_path),
                "model_path": str(args.model_path),
                "data_dir": str(args.data_dir),
                "methods": methods,
                "k": args.k,
                "sim_metric": args.sim_metric,
                "perturb_radius": args.perturb_radius,
                "n_perturb_samples": args.n_perturb_samples,
                "retrieval_pool_size": int(len(retrieval_labels)),
                "retrieval_batch_size": int(args.retrieval_batch_size),
                "scores_path": str(scores_path),
                "summary_path": str(summary_path),
            },
            f,
            indent=2,
        )

    print("✅ Evaluation complete")
    print(f"Per-sample scores: {scores_path}")
    print(f"Method summary: {summary_path}")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Optional: evaluate sanity-check explanations
    # ------------------------------------------------------------------
    if args.sanity_artifact is not None:
        if not args.sanity_artifact.exists():
            raise FileNotFoundError(f"Sanity artifact not found: {args.sanity_artifact}")

        sanity_art = torch.load(args.sanity_artifact, map_location="cpu")
        steps = sanity_art.get("steps", [])
        if not steps:
            raise RuntimeError("Sanity artifact contains no steps.")

        step_idx = args.sanity_step_index if args.sanity_step_index >= 0 else len(steps) + args.sanity_step_index
        if step_idx < 0 or step_idx >= len(steps):
            raise IndexError(f"sanity_step_index {args.sanity_step_index} out of range for {len(steps)} step(s).")
        step = steps[step_idx]
        randomized_layers = step["randomized_layers"]
        step_name = step.get("step_name", f"step_{step_idx}")

        print(f"\nBuilding randomized model for sanity step: {step_name} (layers: {randomized_layers})")
        rand_model = copy.deepcopy(model)
        _randomize_layers(rand_model, randomized_layers, args.seed)
        rand_model.eval()
        rand_explainers = build_explainers(rand_model, methods)

        # Predictions from the randomized model
        with torch.no_grad():
            rand_logits = rand_model(query_images.to(device))
            rand_probs = torch.softmax(rand_logits, dim=1)
            rand_conf, rand_pred = rand_probs.max(dim=1)
        query_pred_labels_sanity = rand_pred.cpu().numpy().astype(int)
        query_confidences_sanity = rand_conf.cpu().numpy().astype(float)

        retrieval_pred_labels_sanity, _ = predict_labels_for_images(
            model=rand_model,
            images=retrieval_images,
            batch_size=args.retrieval_batch_size,
            device=device,
        )

        sanity_all_scores = []
        for method in methods:
            print(f"  Evaluating sanity method: {method}")
            method_scores = evaluate_method(
                method=method,
                explainer=rand_explainers[method],
                query_images=query_images,
                query_labels=query_labels,
                query_pred_labels=query_pred_labels_sanity,
                query_confidences=query_confidences_sanity,
                query_retrieval_indices=query_retrieval_indices,
                retrieval_images=retrieval_images,
                retrieval_labels=retrieval_labels,
                retrieval_pred_labels=retrieval_pred_labels_sanity,
                baseline=baseline,
                k=args.k,
                sim_metric=args.sim_metric,
                perturb_radius=args.perturb_radius,
                n_perturb_samples=args.n_perturb_samples,
                retrieval_batch_size=args.retrieval_batch_size,
                device=device,
            )
            method_scores["sanity_step"] = step_name
            sanity_all_scores.append(method_scores)

        sanity_scores_df = pd.concat(sanity_all_scores, ignore_index=True)
        sanity_summary_df = summarise_scores(sanity_scores_df)

        sanity_scores_path = args.output_dir / "mnist_sanity_explanation_scores.csv"
        sanity_summary_path = args.output_dir / "mnist_sanity_method_summary.csv"
        sanity_scores_df.to_csv(sanity_scores_path, index=False)
        sanity_summary_df.to_csv(sanity_summary_path, index=False)

        print("\n✅ Sanity evaluation complete")
        print(f"Sanity per-sample scores: {sanity_scores_path}")
        print(f"Sanity method summary:    {sanity_summary_path}")
        print(sanity_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
