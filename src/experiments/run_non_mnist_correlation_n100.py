import io
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from correlation_analysis import run_correlation_experiment
from pathlib import Path
from contextlib import redirect_stdout

DATASETS = ["adult", "bank", "beans", "cancer", "heloc", "mushroom", "ocean", "wine"]
rows = []

for ds in DATASETS:
    print(f"Running {ds} with n_samples=100...", flush=True)
    with redirect_stdout(io.StringIO()):
        df = run_correlation_experiment(
            dataset=ds,
            split="test",
            n_samples=100,
            k=10,
            noise_level=0.1,
            seed=42,
            model_name="model1",
            sim_metric="gower",
        )

    ca = pd.to_numeric(df["case_align_R_bounded"], errors="coerce")
    sens = pd.to_numeric(df["captum_sensitivity"], errors="coerce")
    mask = np.isfinite(ca) & np.isfinite(sens)

    if mask.sum() >= 3:
        r, p_r = pearsonr(ca[mask], sens[mask])
        rho, p_rho = spearmanr(ca[mask], sens[mask])
    else:
        r = p_r = rho = p_rho = np.nan

    rows.append({
        "dataset": ds,
        "mean_case_align": float(ca[mask].mean()) if mask.sum() else np.nan,
        "mean_sensitivity": float(sens[mask].mean()) if mask.sum() else np.nan,
        "pearson_r": float(r) if np.isfinite(r) else np.nan,
        "pearson_p": float(p_r) if np.isfinite(p_r) else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_p": float(p_rho) if np.isfinite(p_rho) else np.nan,
        "n": int(mask.sum()),
    })

summary = pd.DataFrame(rows)
out_dir = Path("../results/case_align_consistent_correlation")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "non_mnist_correlation_summary.csv"
summary.to_csv(out_file, index=False)

print("\nDone. Final table:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"\nSaved: {out_file.resolve()}")
