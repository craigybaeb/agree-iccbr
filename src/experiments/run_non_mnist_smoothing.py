import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import NoiseTunnel
from scipy.stats import pearsonr, spearmanr

from correlation_analysis import load_model_and_explainer, load_data
from case_align.case_align import RobustnessCBR

DATASETS = ["adult", "bank", "beans", "cancer", "heloc", "mushroom", "ocean", "wine"]

N_SAMPLES = 100
K = 10
NOISE_LEVEL = 0.1
N_PERTURB = 10
SEED = 42
MODEL_NAME = "model1"
SIM_METRIC = "gower"

NT_TYPE = "smoothgrad"
NT_SAMPLES = 8
NT_STDEVS = 0.1
NOISE_CONTINUOUS = 0.1

rng = np.random.default_rng(SEED)


def infer_continuous_mask(X: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """
    Infer a per-feature mask for continuous columns.

    Columns that are effectively binary ({0,1}) are treated as categorical-like
    and will not receive SmoothGrad noise in categorical-aware mode.
    """
    n_features = X.shape[1]
    continuous_mask = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        col = X[:, j]
        finite_col = col[np.isfinite(col)]
        if finite_col.size == 0:
            continue
        is_binary_like = np.all(np.isclose(finite_col, 0.0, atol=atol) | np.isclose(finite_col, 1.0, atol=atol))
        if is_binary_like:
            continuous_mask[j] = 0.0

    return continuous_mask


def _predict_class(model, x_single: torch.Tensor) -> int:
    with torch.no_grad():
        out = model(x_single)
        return int(torch.argmax(out, dim=1).item())


def _attribute_raw(explainer, x_single: torch.Tensor, target: int) -> np.ndarray:
    a = explainer.attribute(x_single, target=target)
    return a.detach().cpu().numpy().reshape(-1)


def _attribute_smooth(
    explainer,
    x_single: torch.Tensor,
    target: int,
    per_feature_stdevs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute smoothed attributions via manual noise application.
    
    If per_feature_stdevs is provided, apply selective noise per feature.
    Otherwise, use standard NoiseTunnel with uniform noise.
    """
    if per_feature_stdevs is None:
        # Standard approach: use NoiseTunnel with uniform noise
        nt = NoiseTunnel(explainer)
        a = nt.attribute(
            x_single,
            target=target,
            nt_type=NT_TYPE,
            nt_samples=NT_SAMPLES,
            stdevs=NT_STDEVS,
        )
        return a.detach().cpu().numpy().reshape(-1)
    else:
        # Manual smoothing with per-feature selective noise
        x_np = x_single.detach().cpu().numpy().reshape(-1)
        attrs_list = []
        
        # Compute attribution for original input
        a_orig = explainer.attribute(x_single, target=target)
        attrs_list.append(a_orig.detach().cpu().numpy().reshape(-1))
        
        # Generate noisy versions and compute attributions
        for _ in range(NT_SAMPLES):
            # Create per-feature noise: stdevs determine which features get noise
            noise = np.random.normal(0, 1, x_np.shape)
            noise = noise * per_feature_stdevs
            x_noisy_np = x_np + noise
            x_noisy = torch.tensor(x_noisy_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            
            a_noisy = explainer.attribute(x_noisy, target=target)
            attrs_list.append(a_noisy.detach().cpu().numpy().reshape(-1))
        
        # Average attributions across all samples
        attrs_array = np.asarray(attrs_list, dtype=np.float32)
        return attrs_array.mean(axis=0)


def generate_explanations(
    model,
    explainer,
    X: np.ndarray,
    sample_indices: np.ndarray,
    smooth: bool,
    per_feature_stdevs: np.ndarray | None = None,
) -> np.ndarray:
    X_tensor = torch.tensor(X[sample_indices], dtype=torch.float32)
    out = []
    for i, x in enumerate(X_tensor):
        x_single = x.unsqueeze(0).requires_grad_(True)
        target = _predict_class(model, x_single)
        if smooth:
            attr = _attribute_smooth(
                explainer,
                x_single,
                target,
                per_feature_stdevs=per_feature_stdevs,
            )
        else:
            attr = _attribute_raw(explainer, x_single, target)
        out.append(attr)
        if (i + 1) % 20 == 0 or i == len(X_tensor) - 1:
            print(f"      attrs ({'smooth' if smooth else 'raw'}): {i + 1}/{len(X_tensor)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def sensitivity_manual(
    model,
    explainer,
    X: np.ndarray,
    index: int,
    smooth: bool,
    per_feature_stdevs: np.ndarray | None = None,
) -> float:
    x_i = X[index]
    x0 = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    t0 = _predict_class(model, x0)
    if smooth:
        a0 = _attribute_smooth(explainer, x0, t0, per_feature_stdevs=per_feature_stdevs)
    else:
        a0 = _attribute_raw(explainer, x0, t0)

    vals = []
    for _ in range(N_PERTURB):
        xp = x_i + np.random.normal(0, NOISE_LEVEL, x_i.shape)
        xt = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        tp = _predict_class(model, xt)
        if smooth:
            ap = _attribute_smooth(explainer, xt, tp, per_feature_stdevs=per_feature_stdevs)
        else:
            ap = _attribute_raw(explainer, xt, tp)
        vals.append(float(np.linalg.norm(a0 - ap)))
    return float(np.mean(vals))


def corr_safe(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    pr, pp = pearsonr(a[m], b[m])
    sr, sp = spearmanr(a[m], b[m])
    return float(pr), float(pp), float(sr), float(sp)


def main():
    global N_SAMPLES, NT_SAMPLES, NT_STDEVS, NT_TYPE, N_PERTURB, NOISE_CONTINUOUS

    parser = argparse.ArgumentParser(description="Run non-MNIST smoothing vs raw evaluation")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--nt-samples", type=int, default=NT_SAMPLES)
    parser.add_argument("--nt-stdevs", type=float, default=NT_STDEVS)
    parser.add_argument("--nt-type", type=str, default=NT_TYPE)
    parser.add_argument("--n-perturb", type=int, default=N_PERTURB)
    parser.add_argument("--categorical-aware", action="store_true")
    parser.add_argument("--noise-continuous", type=float, default=NOISE_CONTINUOUS)
    parser.add_argument("--skip-mushroom", action="store_true")
    parser.add_argument("--output-suffix", type=str, default="")
    args = parser.parse_args()

    N_SAMPLES = int(args.n_samples)
    NT_SAMPLES = int(args.nt_samples)
    NT_STDEVS = float(args.nt_stdevs)
    NT_TYPE = str(args.nt_type)
    N_PERTURB = int(args.n_perturb)
    NOISE_CONTINUOUS = float(args.noise_continuous)

    all_rows = []
    summary_rows = []
    
    datasets_to_process = DATASETS
    if args.skip_mushroom:
        datasets_to_process = [d for d in DATASETS if d != "mushroom"]
        print(f"Skipping mushroom dataset", flush=True)

    for ds in datasets_to_process:
        print(f"\n=== Dataset: {ds} ===", flush=True)
        np.random.seed(SEED)

        model, explainer = load_model_and_explainer(ds, MODEL_NAME)
        X, y = load_data(ds, split="test")

        per_feature_stdevs = None
        if args.categorical_aware:
            mask_np = infer_continuous_mask(X)
            n_cont = int(mask_np.sum())
            n_cat_like = int(mask_np.size - n_cont)
            print(
                f"   Categorical-aware smoothing enabled: continuous={n_cont}, categorical_like={n_cat_like}",
                flush=True,
            )
            per_feature_stdevs = mask_np * NOISE_CONTINUOUS
            print(f"   Using per-feature stdevs: {n_cont} continuous features get {NOISE_CONTINUOUS}, {n_cat_like} categorical get 0", flush=True)

        n_eval = min(N_SAMPLES, X.shape[0])
        sample_indices = rng.choice(X.shape[0], size=n_eval, replace=False)

        print(f"   Generating raw explanations for {n_eval} samples...", flush=True)
        E_raw = generate_explanations(
            model,
            explainer,
            X,
            sample_indices,
            smooth=False,
            per_feature_stdevs=per_feature_stdevs,
        )

        print(f"   Generating smoothed explanations for {n_eval} samples...", flush=True)
        E_smooth = generate_explanations(
            model,
            explainer,
            X,
            sample_indices,
            smooth=True,
            per_feature_stdevs=per_feature_stdevs,
        )

        print("   Fitting Case Align models...", flush=True)
        cbr_raw = RobustnessCBR(
            k=K,
            m_unlike=1,
            sim_metric=SIM_METRIC,
            problem_metric=SIM_METRIC,
            like_only=True,
            robust_mode="geom",
            random_state=SEED,
        )
        cbr_raw.fit(X[sample_indices], y[sample_indices], E_raw)

        cbr_smooth = RobustnessCBR(
            k=K,
            m_unlike=1,
            sim_metric=SIM_METRIC,
            problem_metric=SIM_METRIC,
            like_only=True,
            robust_mode="geom",
            random_state=SEED,
        )
        cbr_smooth.fit(X[sample_indices], y[sample_indices], E_smooth)

        print("   Computing per-sample metrics (raw/smooth/delta)...", flush=True)
        ds_rows = []
        for pos in range(n_eval):
            idx = int(sample_indices[pos])

            r_raw = cbr_raw.compute_for_index(pos)
            r_smooth = cbr_smooth.compute_for_index(pos)

            sens_raw = sensitivity_manual(
                model,
                explainer,
                X,
                idx,
                smooth=False,
                per_feature_stdevs=per_feature_stdevs,
            )
            sens_smooth = sensitivity_manual(
                model,
                explainer,
                X,
                idx,
                smooth=True,
                per_feature_stdevs=per_feature_stdevs,
            )

            d = E_smooth[pos] - E_raw[pos]
            delta_l1_mean = float(np.mean(np.abs(d)))
            delta_l2 = float(np.linalg.norm(d))
            delta_cos = float(
                np.dot(E_raw[pos], E_smooth[pos])
                / (np.linalg.norm(E_raw[pos]) * np.linalg.norm(E_smooth[pos]) + 1e-8)
            )

            row = {
                "dataset": ds,
                "sample_pos": int(pos),
                "index": idx,
                "class": int(y[idx]),
                "case_align_raw": float(r_raw.R_bounded),
                "case_align_smooth": float(r_smooth.R_bounded),
                "delta_case_align": float(r_smooth.R_bounded - r_raw.R_bounded),
                "sensitivity_raw": float(sens_raw),
                "sensitivity_smooth": float(sens_smooth),
                "delta_sensitivity": float(sens_smooth - sens_raw),
                "delta_expl_l1_mean": delta_l1_mean,
                "delta_expl_l2": delta_l2,
                "expl_cosine_raw_vs_smooth": delta_cos,
            }
            ds_rows.append(row)
            all_rows.append(row)

            if (pos + 1) % 10 == 0 or pos == n_eval - 1:
                print(f"      metrics: {pos + 1}/{n_eval}", flush=True)

        ds_df = pd.DataFrame(ds_rows)

        pr_raw, pp_raw, sr_raw, sp_raw = corr_safe(
            ds_df["case_align_raw"].to_numpy(), ds_df["sensitivity_raw"].to_numpy()
        )
        pr_sm, pp_sm, sr_sm, sp_sm = corr_safe(
            ds_df["case_align_smooth"].to_numpy(), ds_df["sensitivity_smooth"].to_numpy()
        )

        summary_rows.append(
            {
                "dataset": ds,
                "n": int(len(ds_df)),
                "mean_case_align_raw": float(ds_df["case_align_raw"].mean()),
                "mean_case_align_smooth": float(ds_df["case_align_smooth"].mean()),
                "mean_delta_case_align": float(ds_df["delta_case_align"].mean()),
                "mean_sensitivity_raw": float(ds_df["sensitivity_raw"].mean()),
                "mean_sensitivity_smooth": float(ds_df["sensitivity_smooth"].mean()),
                "mean_delta_sensitivity": float(ds_df["delta_sensitivity"].mean()),
                "mean_delta_expl_l1_mean": float(ds_df["delta_expl_l1_mean"].mean()),
                "mean_delta_expl_l2": float(ds_df["delta_expl_l2"].mean()),
                "mean_expl_cosine_raw_vs_smooth": float(ds_df["expl_cosine_raw_vs_smooth"].mean()),
                "pearson_raw_ca_vs_sens": pr_raw,
                "pearson_raw_p": pp_raw,
                "spearman_raw_ca_vs_sens": sr_raw,
                "spearman_raw_p": sp_raw,
                "pearson_smooth_ca_vs_sens": pr_sm,
                "pearson_smooth_p": pp_sm,
                "spearman_smooth_ca_vs_sens": sr_sm,
                "spearman_smooth_p": sp_sm,
            }
        )

    all_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    overall = {
        "datasets": DATASETS,
        "n_total": int(len(all_df)),
        "mean_case_align_raw": float(all_df["case_align_raw"].mean()),
        "mean_case_align_smooth": float(all_df["case_align_smooth"].mean()),
        "mean_delta_case_align": float(all_df["delta_case_align"].mean()),
        "mean_sensitivity_raw": float(all_df["sensitivity_raw"].mean()),
        "mean_sensitivity_smooth": float(all_df["sensitivity_smooth"].mean()),
        "mean_delta_sensitivity": float(all_df["delta_sensitivity"].mean()),
        "mean_delta_expl_l1_mean": float(all_df["delta_expl_l1_mean"].mean()),
        "mean_delta_expl_l2": float(all_df["delta_expl_l2"].mean()),
        "mean_expl_cosine_raw_vs_smooth": float(all_df["expl_cosine_raw_vs_smooth"].mean()),
    }

    out_dir = Path("../results/non_mnist_smoothing_case_align_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.output_suffix.strip()}" if args.output_suffix.strip() else ""

    per_sample_path = out_dir / f"non_mnist_smooth_vs_raw_per_sample{suffix}.csv"
    summary_path = out_dir / f"non_mnist_smooth_vs_raw_summary{suffix}.csv"
    overall_path = out_dir / f"non_mnist_smooth_vs_raw_overall{suffix}.json"

    all_df.to_csv(per_sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n===== SUMMARY (per dataset) =====")
    print(
        summary_df[
            [
                "dataset",
                "n",
                "mean_case_align_raw",
                "mean_case_align_smooth",
                "mean_delta_case_align",
                "mean_sensitivity_raw",
                "mean_sensitivity_smooth",
                "mean_delta_sensitivity",
                "mean_delta_expl_l1_mean",
                "mean_delta_expl_l2",
                "mean_expl_cosine_raw_vs_smooth",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n===== OVERALL =====")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved:")
    print(per_sample_path.resolve())
    print(summary_path.resolve())
    print(overall_path.resolve())


if __name__ == "__main__":
    main()
