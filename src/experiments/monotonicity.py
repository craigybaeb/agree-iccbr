from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR


def _load_split(root: Path, dataset: str, split: str):
    import torch

    x_path = root / "data" / dataset / f"X{split}.pt"
    y_path = root / "data" / dataset / f"y{split}.pt"
    X = torch.load(x_path, map_location="cpu")
    y = torch.load(y_path, map_location="cpu")

    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    else:
        y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] > 1:
        y = y.argmax(axis=1)
    y = y.reshape(-1)

    return X, y


def _load_explanations(root: Path, dataset: str, expl_path: str, X: np.ndarray):
    if expl_path:
        path = Path(expl_path)
    else:
        path = root / "explanations" / dataset / "attributions.npy"

    if path.exists():
        expl = np.load(path, allow_pickle=True)
        if not hasattr(expl, "shape") or expl.shape[0] != X.shape[0]:
            raise ValueError(f"Explanations at {path} do not align with X shape {X.shape}")
        if expl.ndim > 2:
            expl = expl.reshape(expl.shape[0], -1)
        return expl, str(path)

    return X.copy(), "<X>"


def main():
    parser = argparse.ArgumentParser(description="Monotonicity experiment (real data)")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expl_path", type=str, default="")
    parser.add_argument("--out", type=str, default="results/monotonicity.csv")
    args = parser.parse_args()

    root = ROOT
    X, y = _load_split(root, args.dataset, args.split)
    expl, expl_src = _load_explanations(root, args.dataset, args.expl_path, X)

    cbr = RobustnessCBR(k=5, m_unlike=1, sim_metric="gower", problem_metric="gower").fit(
        X, y, explanations=expl
    )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=n, replace=False)

    rows: List[dict] = []
    for i in sample_idx:
        res = cbr.compute_for_index(int(i))
        rows.append(
            {
                "index": int(i),
                "S_plus": float(res.S_plus),
                "S_minus_u": float(res.S_minus_u),
                "S_minus_x": float(res.S_minus_x),
                "one_minus_S_minus_x": float(1.0 - res.S_minus_x),
                "R_bounded": float(res.R_bounded),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
