from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _ensure_repo_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _try_import_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


TORCH_AVAILABLE = _try_import_torch()


def _as_tensor(X):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for explanations.")
    import torch

    if isinstance(X, torch.Tensor):
        return X.float()
    return torch.tensor(np.asarray(X), dtype=torch.float32)


def _load_dataset(root: Path, dataset: str, split: str = "test"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for explanations.")
    import torch

    data_dir = root / "data" / dataset
    x_path = data_dir / f"X{split}.pt"
    y_path = data_dir / f"y{split}.pt"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_path} or {y_path}")
    X = torch.load(x_path, map_location="cpu")
    y = torch.load(y_path, map_location="cpu")
    return X, y


def _baseline_for_dataset(root: Path, dataset: str, X_test):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for explanations.")
    import torch

    try:
        X_train = torch.load(root / "data" / dataset / "Xtrain.pt", map_location="cpu").float()
        baseline = X_train.mean(dim=0, keepdim=True)
    except Exception:
        baseline = torch.zeros(1, X_test.shape[1])
    return baseline


def _load_model(root: Path, dataset: str, model_name: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for explanations.")
    import torch
    from load.nets import training as net_training

    model_path = root / "models" / f"{dataset}_{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model_dict = net_training.model_dict(dataset)
    if model_name not in model_dict:
        raise KeyError(f"Model name {model_name} not found in model_dict. Available: {list(model_dict.keys())}")

    model = model_dict[model_name]
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def explain_batch(
    model,
    X,
    methods: Sequence[str] = ("lrp", "dl", "ig"),
    baselines=None,
    batch_size: int = 256,
) -> Dict[str, "torch.Tensor"]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for explanations.")
    import torch
    from captum.attr import DeepLift, IntegratedGradients
    from explainers.lrp import LRP

    X = _as_tensor(X)
    n = X.shape[0]
    results = {}

    for method in methods:
        if method == "lrp":
            explainer = LRP(model)
        elif method == "dl":
            explainer = DeepLift(model)
        elif method == "ig":
            explainer = IntegratedGradients(model)
        else:
            raise ValueError(f"Unknown method: {method}")

        attrs = []
        for i in range(0, n, batch_size):
            xb = X[i : i + batch_size].clone().requires_grad_(True)
            with torch.no_grad():
                logits = model(xb)
            targets = logits.argmax(dim=1)

            if method in ("dl", "ig"):
                b = baselines
                if b is None:
                    b = torch.zeros(1, xb.shape[1])
                if b.shape[0] == 1:
                    b = b.expand_as(xb)
                attr = explainer.attribute(xb, baselines=b, target=targets)
            else:
                attr = explainer.attribute(xb, target=targets)

            attrs.append(attr.detach().cpu())

        results[method] = torch.cat(attrs, dim=0)

    return results


def explain_dataset(
    dataset: str,
    model_name: str = "model1",
    split: str = "test",
    methods: Sequence[str] = ("lrp", "dl", "ig"),
    batch_size: int = 256,
    out_dir: Optional[Path] = None,
) -> Dict[str, str]:
    root = _ensure_repo_on_path()
    if out_dir is None:
        out_dir = root / "explainers"

    X, _ = _load_dataset(root, dataset, split=split)
    X = _as_tensor(X)
    model = _load_model(root, dataset, model_name)
    baseline = _baseline_for_dataset(root, dataset, X)

    attrs = explain_batch(model, X, methods=methods, baselines=baseline, batch_size=batch_size)

    out_dir = Path(out_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for method, attr in attrs.items():
        save_path = out_dir / f"{dataset}_{model_name}_{split}_{method}.pt"
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to save explanations.")
        import torch

        torch.save(attr, save_path)
        saved[method] = str(save_path)

    return saved


def explain_all_datasets(
    datasets: Iterable[str],
    model_name: str = "model1",
    split: str = "test",
    methods: Sequence[str] = ("lrp", "dl", "ig"),
    batch_size: int = 256,
    out_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for dataset in datasets:
        results[dataset] = explain_dataset(
            dataset,
            model_name=model_name,
            split=split,
            methods=methods,
            batch_size=batch_size,
            out_dir=out_dir,
        )
    return results
