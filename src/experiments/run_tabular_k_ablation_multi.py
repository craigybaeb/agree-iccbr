#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_align.case_align import RobustnessCBR
from correlation_analysis import load_data

DEFAULT_DATASETS = ["adult", "bank", "beans", "cancer", "heloc", "ocean", "wine"]
DEFAULT_K_VALUES = list(range(3, 22, 2)) + list(range(25, 50, 4))
DEFAULT_SEED = 42


def parse_k_values(text: str) -> List[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values:
        raise ValueError("At least one k value is required")
    return values


def collect_attribution_variants(
    attributions_root: Path,
    dataset: str,
    split: str,
    variant_suffix: str | None = None,
) -> List[Tuple[str, Path]]:
    pattern = f"{dataset}_model*/{split}/attributions.npy"
    files = sorted(attributions_root.glob(pattern))

    variants = []
    for p in files:
        # label like adult_model1
        label = p.parents[1].name
        if variant_suffix is not None and not label.endswith(variant_suffix):
            continue
        variants.append((label, p))
    return variants


def choose_raw_variant(variants: List[Tuple[str, Path]], preferred_raw: str | None) -> Tuple[str, Path]:
    if not variants:
        raise ValueError("No attribution variants available")

    if preferred_raw is not None:
        preferred_raw = preferred_raw.strip()
        for label, path in variants:
            if label == preferred_raw:
                return label, path

    for label, path in variants:
        if label.endswith("model1"):
            return label, path

    return variants[0]


def compute_curve(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    E_sub: np.ndarray,
    k_values: List[int],
    sim_metric: str,
    seed: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}

    for k in k_values:
        cbr = RobustnessCBR(
            k=int(k),
            m_unlike=1,
            sim_metric=sim_metric,
            problem_metric=sim_metric,
            like_only=True,
            robust_mode="geom",
            random_state=seed,
        )
        cbr.fit(X_sub, y_sub, E_sub)

        vals = np.asarray([float(cbr.compute_for_index(i).R_bounded) for i in range(X_sub.shape[0])], dtype=np.float32)
        out[int(k)] = vals

    return out


def make_paper_ready_layout(fig: go.Figure):
    fig.update_layout(
        title=None,
        template="plotly_white",
        width=1200,
        height=700,
        font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title="",
            font=dict(size=14),
        ),
        margin=dict(l=70, r=20, t=90, b=70),
    )

    fig.update_xaxes(
        title="k",
        showline=True,
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        ticklen=6,
        showgrid=True,
        gridwidth=0.8,
    )
    fig.update_yaxes(
        title="CaseAlign",
        showline=True,
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        ticklen=6,
        showgrid=True,
        gridwidth=0.8,
    )


def save_plotly_figure(fig: go.Figure, output_html: Path, output_png: Path) -> bool:
    fig.write_html(output_html)
    try:
        fig.write_image(output_png, scale=3)
        return True
    except Exception as exc:
        print(f"Warning: PNG export skipped ({exc}). HTML was saved.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run tabular k-ablation from SAVED attributions only (no explainer generation). "
            "Builds one raw figure and one variant with error bars from attribution variation."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sim-metric", type=str, default="gower")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--attributions-root", type=str, default="explanations/results_medoid")
    parser.add_argument(
        "--raw-variant-suffix",
        type=str,
        default="model1",
        help="Preferred raw line variant suffix (e.g. model1). Falls back automatically if unavailable.",
    )
    parser.add_argument("--output-dir", type=str, default="results/k_ablation_multi_dataset")
    parser.add_argument("--output-prefix", type=str, default="tabular_all_excl_mushroom_from_saved_attr")
    parser.add_argument(
        "--variant-suffix-filter",
        type=str,
        default=None,
        help="Optional suffix filter for attribution variants, e.g. model1 to use only *_model1 folders.",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    rng = np.random.default_rng(int(args.seed))

    attributions_root = (ROOT / args.attributions_root).resolve()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Multi-dataset K-ablation from saved attributions")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Split: {args.split}")
    print(f"k values: {k_values}")
    print(f"Attributions root: {attributions_root}")
    if args.variant_suffix_filter:
        print(f"Variant suffix filter: {args.variant_suffix_filter}")

    per_sample_rows = []
    summary_rows = []

    for dataset in args.datasets:
        print(f"\n--- Dataset: {dataset} ---", flush=True)
        variants = collect_attribution_variants(
            attributions_root,
            dataset,
            args.split,
            variant_suffix=args.variant_suffix_filter,
        )
        if not variants:
            print(f"Warning: no saved attribution variants found for dataset={dataset}, split={args.split}; skipped.")
            continue

        preferred_raw_full = f"{dataset}_{args.raw_variant_suffix}" if args.raw_variant_suffix else None
        raw_label, raw_path = choose_raw_variant(variants, preferred_raw_full)
        print(f"Using raw variant: {raw_label}")
        print(f"Found variants: {[label for label, _ in variants]}")

        X, y = load_data(dataset, split=args.split)
        n_eval = min(int(args.n_samples), X.shape[0])
        sample_idx = rng.choice(X.shape[0], size=n_eval, replace=False)
        X_sub = X[sample_idx].astype(np.float32)
        y_sub = y[sample_idx].astype(int)

        print(f"Using {n_eval} sampled points")

        for variant_label, variant_path in variants:
            E = np.load(variant_path)
            if E.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Attribution rows mismatch for {variant_path}: E has {E.shape[0]} rows, X has {X.shape[0]}"
                )

            E_sub = E[sample_idx].astype(np.float32)
            curve = compute_curve(
                X_sub=X_sub,
                y_sub=y_sub,
                E_sub=E_sub,
                k_values=k_values,
                sim_metric=args.sim_metric,
                seed=int(args.seed),
            )

            for k, values in curve.items():
                for pos, val in enumerate(values):
                    per_sample_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant_label,
                            "k": int(k),
                            "sample_pos": int(pos),
                            "R_bounded": float(val),
                        }
                    )

                summary_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant_label,
                        "k": int(k),
                        "mean_R_bounded": float(np.mean(values)),
                        "std_R_bounded": float(np.std(values)),
                        "n_samples": int(values.shape[0]),
                        "is_raw_variant": int(variant_label == raw_label),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("No dataset had usable saved attribution files. Nothing to plot.")

    per_sample_df = pd.DataFrame(per_sample_rows)

    raw_df = summary_df[summary_df["is_raw_variant"] == 1].copy()

    var_df = (
        summary_df.groupby(["dataset", "k"], as_index=False)
        .agg(
            mean_R_bounded_across_variants=("mean_R_bounded", "mean"),
            std_R_bounded_across_variants=("mean_R_bounded", "std"),
            n_variants=("mean_R_bounded", "count"),
        )
        .fillna({"std_R_bounded_across_variants": 0.0})
    )

    fig_raw = go.Figure()
    for dataset in args.datasets:
        ds = raw_df[raw_df["dataset"] == dataset].sort_values("k")
        if ds.empty:
            continue
        fig_raw.add_trace(
            go.Scatter(
                x=ds["k"],
                y=ds["mean_R_bounded"],
                mode="lines+markers",
                name=dataset.capitalize(),
                line=dict(width=3),
                marker=dict(size=8),
            )
        )

    make_paper_ready_layout(fig_raw)

    fig_var = go.Figure()
    for dataset in args.datasets:
        ds = var_df[var_df["dataset"] == dataset].sort_values("k")
        if ds.empty:
            continue
        fig_var.add_trace(
            go.Scatter(
                x=ds["k"],
                y=ds["mean_R_bounded_across_variants"],
                mode="lines+markers",
                name=dataset.capitalize(),
                line=dict(width=3),
                marker=dict(size=8),
                error_y=dict(
                    type="data",
                    array=ds["std_R_bounded_across_variants"],
                    visible=True,
                    thickness=1.5,
                    width=4,
                ),
            )
        )

    make_paper_ready_layout(fig_var)

    per_sample_path = out_dir / f"{args.output_prefix}_per_sample.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary_by_variant.csv"
    var_path = out_dir / f"{args.output_prefix}_summary_across_variants.csv"

    raw_html = out_dir / f"{args.output_prefix}_raw_no_errorbars.html"
    raw_png = out_dir / f"{args.output_prefix}_raw_no_errorbars.png"
    var_html = out_dir / f"{args.output_prefix}_variation_errorbars.html"
    var_png = out_dir / f"{args.output_prefix}_variation_errorbars.png"

    per_sample_df.to_csv(per_sample_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    var_df.to_csv(var_path, index=False)

    raw_png_ok = save_plotly_figure(fig_raw, raw_html, raw_png)
    var_png_ok = save_plotly_figure(fig_var, var_html, var_png)

    print("\nSaved outputs:")
    print(per_sample_path)
    print(summary_path)
    print(var_path)
    print(raw_html)
    print(var_html)
    if raw_png_ok:
        print(raw_png)
    if var_png_ok:
        print(var_png)


if __name__ == "__main__":
    main()
