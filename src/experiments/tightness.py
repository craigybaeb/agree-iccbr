from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR
from case_align.metrics import gower_distance, sim_to_dist


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


def _align_local_values(
    cbr: RobustnessCBR, i: int, neigh_expl: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    if neigh_expl.shape[0] == 0:
        return np.array([], dtype=float)
    dsoln_neigh = np.array([cbr._solution_distance(cbr.expl[i], v) for v in neigh_expl])  # noqa: SLF001
    dsmin = float(np.min(dsoln_neigh))
    dsmax = float(np.max(dsoln_neigh))
    denom = max(dsmax - dsmin, eps)
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    return align


def _case_align_weighted_indices(
    cbr: RobustnessCBR, t_idx: int, neigh_idx: np.ndarray, eps: float = 1e-8
) -> float:
    if neigh_idx.size == 0:
        return 0.0
    dprob = cbr._problem_dists_to(t_idx)[neigh_idx]  # noqa: SLF001
    weights = 1.0 - dprob
    # Use global min/max distance (definition of Align)
    dsoln_all = np.array(
        [cbr._solution_distance(cbr.expl[t_idx], cbr.expl[j]) for j in range(cbr.expl.shape[0])]
    )
    dsoln_neigh = dsoln_all[neigh_idx]
    dsmin = float(np.min(dsoln_all))
    dsmax = float(np.max(dsoln_all))
    denom = max(dsmax - dsmin, eps)
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return 0.0
    return float(np.sum(weights * align) / wsum)


def _like_indices(y: np.ndarray, i: int):
    mask = y == y[i]
    mask[i] = False
    return np.where(mask)[0]


def _unlike_indices(y: np.ndarray, i: int):
    return np.where(y != y[i])[0]


def _far_neigh_idx(cbr: RobustnessCBR, i: int, candidate_idx: np.ndarray, k: int):
    # Use explanation-space distances (solution space) to pick farthest neighbors.
    dsoln = np.array([cbr._solution_distance(cbr.expl[i], cbr.expl[j]) for j in candidate_idx])  # noqa: SLF001
    order = np.argsort(-dsoln)
    sel = candidate_idx[order[: min(k, candidate_idx.size)]]
    return sel


def _farthest_unlike_anchor(cbr: RobustnessCBR, i: int, y: np.ndarray):
    # Use problem-space distances to pick farthest unlike anchor.
    unlike_idx = _unlike_indices(y, i)
    if unlike_idx.size == 0:
        return None
    dprob = cbr._problem_dists_to(i)  # noqa: SLF001
    far_idx = unlike_idx[np.argmax(dprob[unlike_idx])]
    return int(far_idx)


def _rand_neigh_idx(rng: np.random.Generator, candidate_idx: np.ndarray, k: int):
    if candidate_idx.size == 0:
        return candidate_idx
    size = min(k, candidate_idx.size)
    return rng.choice(candidate_idx, size=size, replace=False)


def _noise_vectors(rng: np.random.Generator, mean: np.ndarray, std: np.ndarray, k: int):
    return rng.normal(loc=mean, scale=std, size=(k, mean.shape[0]))


def _align_with_noise(
    cbr: RobustnessCBR,
    i: int,
    noise_vecs: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Alignment using synthetic explanation vectors with uniform weights.
    We keep normalization (dsmin/dsmax) based on real explanations.
    """
    if cbr.expl is None:
        raise RuntimeError("CBR must be fit with explanations.")
    if noise_vecs.shape[0] == 0:
        return 0.0
    dsoln_all = np.array(
        [cbr._solution_distance(cbr.expl[i], cbr.expl[j]) for j in range(cbr.expl.shape[0])]
    )
    dsmin = float(np.min(dsoln_all))
    dsmax = float(np.max(dsoln_all))
    denom = max(dsmax - dsmin, eps)
    dsoln_neigh = np.array([cbr._solution_distance(cbr.expl[i], v) for v in noise_vecs])
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    return float(np.mean(align)) if align.size else 0.0


def _case_align_weighted_noise(
    cbr: RobustnessCBR,
    t_idx: int,
    noise_x: np.ndarray,
    noise_expl: np.ndarray,
    eps: float = 1e-8,
) -> float:
    if noise_x.shape[0] == 0:
        return 0.0
    if cbr.problem_metric == "gower":
        dprob = np.array([gower_distance(cbr.X[t_idx], v, cbr._feature_ranges_used) for v in noise_x])  # noqa: SLF001
    else:
        # Default to cosine similarity mapped to distance
        xi = cbr.X[t_idx]
        denom = (np.linalg.norm(noise_x, axis=1) * np.linalg.norm(xi)) + cbr.epsilon
        sim = (noise_x @ xi) / denom
        dprob = np.array([sim_to_dist(float(s)) for s in sim])
    weights = 1.0 - dprob
    dsoln_all = np.array(
        [cbr._solution_distance(cbr.expl[t_idx], cbr.expl[j]) for j in range(cbr.expl.shape[0])]
    )
    dsmin = float(np.min(dsoln_all))
    dsmax = float(np.max(dsoln_all))
    denom = max(dsmax - dsmin, eps)
    dsoln_neigh = np.array([cbr._solution_distance(cbr.expl[t_idx], v) for v in noise_expl])
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return 0.0
    return float(np.sum(weights * align) / wsum)


def _compute_geom(S_plus: float, S_minus_u: float, S_minus_x: float, eps: float = 1e-8) -> float:
    a = max(S_plus, eps)
    b = max(S_minus_u, eps)
    c = 1.0 - max(S_minus_x, eps)
    return float((a * b * c) ** (1.0 / 3.0))


def run_experiment(
    dataset: str,
    split: str,
    n_samples: int,
    seed: int,
    expl_path: str,
    debug_n: int = 0,
):
    root = ROOT
    print(f"[tightness] dataset={dataset} split={split} n_samples={n_samples} seed={seed}")
    X, y = _load_split(root, dataset, split)
    expl, expl_src = _load_explanations(root, dataset, expl_path, X)
    print(f"[tightness] X shape={X.shape} y shape={y.shape}")
    print(f"[tightness] explanations source={expl_src} shape={expl.shape}")

    cbr = RobustnessCBR(
        k=5,
        m_unlike=1,
        sim_metric="gower",
        problem_metric="gower",
        robust_mode="geom",
    ).fit(X, y, explanations=expl)

    expl_mean = np.mean(expl, axis=0)
    expl_std = np.std(expl, axis=0)
    expl_std = np.where(expl_std == 0, 1e-6, expl_std)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std == 0, 1e-6, X_std)

    rng = np.random.default_rng(seed)
    n = min(n_samples, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=n, replace=False)
    print(f"[tightness] sampling {n} indices for evaluation")

    rows: List[Dict] = []

    for j, i in enumerate(sample_idx, start=1):
        like_idx_real, _ = cbr._neighbours_like(i)  # noqa: SLF001
        like_candidates_x = _like_indices(y, i)
        nun_idx, nun_d = cbr._nearest_unlikes(i)  # noqa: SLF001

        scenario_neigh_x = {
            "real": like_idx_real,
            "identical": np.array([i] * max(1, cbr.k), dtype=int),
            "random_like": _rand_neigh_idx(rng, like_candidates_x, cbr.k),
            "farthest_like": _far_neigh_idx(cbr, i, like_candidates_x, cbr.k),
        }

        scenarios = list(scenario_neigh_x.keys()) + ["random_noise"]

        for scenario in scenarios:
            # S_plus
            if scenario == "random_noise":
                noise_x = _noise_vectors(rng, X_mean, X_std, max(1, cbr.k))
                noise_e = _noise_vectors(rng, expl_mean, expl_std, max(1, cbr.k))
                S_plus_val = _case_align_weighted_noise(cbr, i, noise_x, noise_e, eps=cbr.epsilon)
            else:
                neigh_idx = scenario_neigh_x[scenario]
                S_plus_val = _case_align_weighted_indices(cbr, i, neigh_idx, eps=cbr.epsilon)

            # S_minus_u and S_minus_x (weighted across NUNs)
            if scenario == "identical":
                far_anchor = _farthest_unlike_anchor(cbr, i, y)
                if far_anchor is None:
                    S_minus_u = 0.0
                    S_minus_x = 0.0
                else:
                    neigh_idx = np.array([far_anchor] * max(1, cbr.k), dtype=int)
                    # u_vs_u uses u's identical neighbors
                    S_minus_u = float(_case_align_weighted_indices(cbr, far_anchor, neigh_idx, eps=cbr.epsilon))
                    # x_vs_u_like uses the same identical-to-u neighbors
                    S_minus_x = float(_case_align_weighted_indices(cbr, i, neigh_idx, eps=cbr.epsilon))
            elif nun_idx.size == 0:
                S_minus_u = 0.0
                S_minus_x = 0.0
            else:
                aw = cbr._weight_from_dists(nun_d)  # noqa: SLF001
                S_minus_u_list = []
                S_minus_x_list = []
                for a in nun_idx:
                    like_candidates_u = _like_indices(y, a)
                    if scenario == "random_noise":
                        noise_x = _noise_vectors(rng, X_mean, X_std, max(1, cbr.k))
                        noise_e = _noise_vectors(rng, expl_mean, expl_std, max(1, cbr.k))
                        val_u = _case_align_weighted_noise(cbr, a, noise_x, noise_e, eps=cbr.epsilon)
                        val_x = _case_align_weighted_noise(cbr, i, noise_x, noise_e, eps=cbr.epsilon)
                    else:
                        neigh_idx = (
                            cbr._neighbours_like_of_anchor(a)[0]  # noqa: SLF001
                            if scenario == "real"
                            else np.array([a] * max(1, cbr.k), dtype=int)
                            if scenario == "identical"
                            else _rand_neigh_idx(rng, like_candidates_u, cbr.k)
                            if scenario == "random_like"
                            else _far_neigh_idx(cbr, a, like_candidates_u, cbr.k)
                        )
                        val_u = _case_align_weighted_indices(cbr, a, neigh_idx, eps=cbr.epsilon)
                        val_x = _case_align_weighted_indices(cbr, i, neigh_idx, eps=cbr.epsilon)

                    S_minus_u_list.append(val_u)
                    S_minus_x_list.append(val_x)

                S_minus_u = float(np.sum(aw * np.array(S_minus_u_list)))
                S_minus_x = float(np.sum(aw * np.array(S_minus_x_list)))

            R_bounded = _compute_geom(S_plus_val, S_minus_u, S_minus_x, eps=cbr.epsilon)

            rows.append(
                {
                    "index": int(i),
                    "scenario": scenario,
                    "S_plus": float(S_plus_val),
                    "S_minus_u": float(S_minus_u),
                    "S_minus_x": float(S_minus_x),
                    "R_bounded": float(R_bounded),
                }
            )

            if debug_n > 0 and j <= debug_n and scenario == "identical":
                print("[tightness:debug] index", int(i), "scenario", scenario)
                print("  S_plus=", S_plus_val, "S_minus_u=", S_minus_u, "S_minus_x=", S_minus_x, "R_bounded=", R_bounded)
                if nun_idx.size > 0:
                    print("  nun_idx=", nun_idx.tolist())

        if j % max(1, n // 10) == 0 or j == n:
            print(f"[tightness] progress: {j}/{n}")

    df = pd.DataFrame(rows)
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tightness_{dataset}_{split}.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    summary = (
        df.groupby(["scenario"])
        .agg(
            n=("R_bounded", "count"),
            min=("R_bounded", "min"),
            max=("R_bounded", "max"),
            mean=("R_bounded", "mean"),
            p01=("R_bounded", lambda s: float(np.quantile(s, 0.01))),
            p05=("R_bounded", lambda s: float(np.quantile(s, 0.05))),
            p50=("R_bounded", lambda s: float(np.quantile(s, 0.50))),
            p95=("R_bounded", lambda s: float(np.quantile(s, 0.95))),
            p99=("R_bounded", lambda s: float(np.quantile(s, 0.99))),
        )
        .reset_index()
    )
    summary_path = out_dir / f"tightness_{dataset}_{split}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Tightness experiment")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expl_path", type=str, default="")
    parser.add_argument("--debug_n", type=int, default=0)
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        split=args.split,
        n_samples=args.n_samples,
        seed=args.seed,
        expl_path=args.expl_path,
        debug_n=args.debug_n,
    )


if __name__ == "__main__":
    main()
