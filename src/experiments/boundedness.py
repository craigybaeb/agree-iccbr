from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR


class RobustnessCBRCosineSimilarity(RobustnessCBR):
    """
    Use raw cosine similarity in [-1, 1] for both problem and solution space.
    This intentionally does NOT map to [0,1] so we can test boundedness
    without enforcing [0,1] distances.
    """

    def _problem_dists_to(self, i: int) -> np.ndarray:
        X = self.X
        if X is None:
            raise RuntimeError("Call fit() first.")
        xi = X[i]
        denom = (np.linalg.norm(X, axis=1) * np.linalg.norm(xi)) + self.epsilon
        sim = (X @ xi) / denom
        return sim.astype(float)

    def _solution_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(-1)
        b = b.reshape(-1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + self.epsilon
        return float(np.dot(a, b) / denom)


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


def _bounds_flags(S_plus, S_minus, R_bounded):
    def _out(x):
        return (x < 0.0) | (x > 1.0)

    return _out(S_plus), _out(S_minus), _out(R_bounded)


def main():
    parser = argparse.ArgumentParser(description="Boundedness experiment")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expl_path", type=str, default="")
    args = parser.parse_args()

    root = ROOT
    print(f"[boundedness] dataset={args.dataset} split={args.split} n_samples={args.n_samples} seed={args.seed}")
    X, y = _load_split(root, args.dataset, args.split)
    print(f"[boundedness] loaded X shape={X.shape} y shape={y.shape}")

    # Use provided explanations if available; otherwise fall back to X.
    expl = None
    if args.expl_path:
        expl_path = Path(args.expl_path)
    else:
        expl_path = root / "explanations" / args.dataset / "attributions.npy"

    if expl_path.exists():
        expl = np.load(expl_path, allow_pickle=True)
        if not hasattr(expl, "shape") or expl.shape[0] != X.shape[0]:
            raise ValueError(f"Explanations at {expl_path} do not align with X shape {X.shape}")
        print(f"[boundedness] loaded explanations from {expl_path} shape={expl.shape}")
    else:
        expl = X.copy()
        print("[boundedness] no explanations found; using X as explanations")

    # Flatten explanations to 2D if needed (e.g., images/attributions with extra dims)
    if expl.ndim > 2:
        expl = expl.reshape(expl.shape[0], -1)
        print(f"[boundedness] flattened explanations to shape={expl.shape}")

    print("[boundedness] fitting Gower model...")
    gower = RobustnessCBR(
        k=5,
        m_unlike=1,
        sim_metric="gower",
        problem_metric="gower",
        robust_mode="geom",
    ).fit(X, y, explanations=expl)

    print("[boundedness] fitting Cosine model...")
    cosine = RobustnessCBRCosineSimilarity(
        k=5,
        m_unlike=1,
        sim_metric="cosine",
        problem_metric="cosine",
        robust_mode="geom",
    ).fit(X, y, explanations=expl)

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=n, replace=False)
    print(f"[boundedness] sampling {n} indices for evaluation")

    results_gower = []
    results_cosine = []
    for j, i in enumerate(sample_idx, start=1):
        results_gower.append(gower.compute_for_index(i))
        results_cosine.append(cosine.compute_for_index(i))
        if j % max(1, n // 10) == 0 or j == n:
            print(f"[boundedness] progress: {j}/{n}")

    def to_rows(results: List, metric_name: str):
        rows = []
        for r in results:
            rows.append(
                {
                    "metric": metric_name,
                    "index": r.index,
                    "S_plus": float(r.S_plus),
                    "S_minus": float(r.S_minus),
                    "R_bounded": float(r.R_bounded),
                }
            )
        return rows

    rows = []
    rows.extend(to_rows(results_gower, "gower"))
    rows.extend(to_rows(results_cosine, "cosine_sim"))

    import pandas as pd

    df = pd.DataFrame(rows)
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"boundedness_{args.dataset}_{args.split}.csv"
    df.to_csv(out_path, index=False)

    S_plus = df["S_plus"].to_numpy()
    S_minus = df["S_minus"].to_numpy()
    R_bounded = df["R_bounded"].to_numpy()
    v1, v2, v3 = _bounds_flags(S_plus, S_minus, R_bounded)
    df["violation_S_plus"] = v1
    df["violation_S_minus"] = v2
    df["violation_R_bounded"] = v3
    df["violation_any"] = v1 | v2 | v3
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby("metric")
        .agg(
            n=("index", "count"),
            v_any=("violation_any", "sum"),
            v_S_plus=("violation_S_plus", "sum"),
            v_S_minus=("violation_S_minus", "sum"),
            v_R_bounded=("violation_R_bounded", "sum"),
        )
        .reset_index()
    )
    summary["pct_any"] = summary["v_any"] / summary["n"]
    summary["pct_S_plus"] = summary["v_S_plus"] / summary["n"]
    summary["pct_S_minus"] = summary["v_S_minus"] / summary["n"]
    summary["pct_R_bounded"] = summary["v_R_bounded"] / summary["n"]
    summary_path = out_dir / f"boundedness_{args.dataset}_{args.split}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:", out_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
