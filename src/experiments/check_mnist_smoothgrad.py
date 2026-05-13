from pathlib import Path
import sys

import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mnist_explain_predictions import load_mnist_model, compute_baseline


def total_variation(a: np.ndarray) -> np.ndarray:
    dx = np.diff(a, axis=2)
    dy = np.diff(a, axis=1)
    return np.mean(np.abs(dx), axis=(1, 2)) + np.mean(np.abs(dy), axis=(1, 2))


def laplacian_energy(a: np.ndarray) -> np.ndarray:
    center = a
    up = np.roll(center, 1, axis=1)
    down = np.roll(center, -1, axis=1)
    left = np.roll(center, 1, axis=2)
    right = np.roll(center, -1, axis=2)
    lap = up + down + left + right - 4 * center
    return np.mean(lap**2, axis=(1, 2))


def high_frequency_ratio(a: np.ndarray, cutoff: float = 0.35) -> np.ndarray:
    h, w = a.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / rr.max()
    mask = rr >= cutoff

    out = []
    for x in a:
        fx = np.fft.fftshift(np.fft.fft2(x))
        power = np.abs(fx) ** 2
        out.append(power[mask].sum() / (power.sum() + 1e-12))
    return np.array(out)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = []
    for va, vb in zip(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)):
        cos.append(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return np.array(cos)


def main() -> None:
    root = ROOT
    model_path = root / "src/models/mnist/mnist_best_model.pt"
    artifact_path = root / "src/explanations/mnist/mnist_explanations.pt"
    data_dir = root / "src/data"

    device = torch.device("cpu")
    nt_samples = 32
    nt_stdevs = 0.2
    n = 64

    artifact = torch.load(artifact_path, map_location=device)
    images = artifact["images"][:n].to(device)
    preds = artifact["pred_labels"][:n].to(device)

    model = load_mnist_model(model_path, device)
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    baseline = compute_baseline(data_dir, baseline_type="mean", device=device).expand_as(images)

    raw = ig.attribute(
        images.clone().detach().requires_grad_(True),
        baselines=baseline,
        target=preds,
    ).detach().cpu().numpy()[:, 0]

    smooth = nt.attribute(
        images.clone().detach().requires_grad_(True),
        baselines=baseline,
        target=preds,
        nt_type="smoothgrad",
        nt_samples=nt_samples,
        stdevs=nt_stdevs,
    ).detach().cpu().numpy()[:, 0]

    raw_tv, smooth_tv = total_variation(raw), total_variation(smooth)
    raw_lap, smooth_lap = laplacian_energy(raw), laplacian_energy(smooth)
    raw_hf, smooth_hf = high_frequency_ratio(raw), high_frequency_ratio(smooth)
    cos = cosine_similarity(raw, smooth)

    print(f"N={n}")
    print(f"SmoothGrad: nt_samples={nt_samples}, stdevs={nt_stdevs}")
    print(f"Mean cosine(raw,smooth): {cos.mean():.6f}")

    print("\nMean roughness metrics (lower = smoother):")
    print(f"TV:        raw={raw_tv.mean():.6f}, smooth={smooth_tv.mean():.6f}, delta={((smooth_tv - raw_tv).mean()):.6f}")
    print(f"Laplacian: raw={raw_lap.mean():.6f}, smooth={smooth_lap.mean():.6f}, delta={((smooth_lap - raw_lap).mean()):.6f}")
    print(f"HF-ratio:  raw={raw_hf.mean():.6f}, smooth={smooth_hf.mean():.6f}, delta={((smooth_hf - raw_hf).mean()):.6f}")

    print("\nFraction of samples smoother after SmoothGrad:")
    print(f"TV lower:        {(smooth_tv < raw_tv).mean():.3f}")
    print(f"Laplacian lower: {(smooth_lap < raw_lap).mean():.3f}")
    print(f"HF-ratio lower:  {(smooth_hf < raw_hf).mean():.3f}")


if __name__ == "__main__":
    main()
