from typing import Optional

import numpy as np


def safe_normalise_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise rows; safe for zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between 1D vectors in [-1, 1]."""
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + eps
    return float(np.dot(vec_a, vec_b) / denom)


def rankdata(vec: np.ndarray) -> np.ndarray:
    """Average rank for ties, 1..n."""
    temp = vec.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(vec), dtype=float)
    uniq, inv, counts = np.unique(vec, return_inverse=True, return_counts=True)
    cumsum = np.cumsum(counts)
    starts = cumsum - counts
    avg_ranks = (starts + cumsum - 1) / 2.0
    return avg_ranks[inv] + 1.0


def spearman_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """Spearman rank correlation in [-1, 1]."""
    ranks_a = rankdata(vec_a)
    ranks_b = rankdata(vec_b)
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    denom = (np.linalg.norm(ranks_a) * np.linalg.norm(ranks_b)) + eps
    return float(np.dot(ranks_a, ranks_b) / denom)


def sim_to_dist(similarity: float) -> float:
    """Map similarity in [-1, 1] to distance in [0, 1]."""
    sim_01 = 0.5 * (similarity + 1.0)
    return float(1.0 - sim_01)


def gower_distance_matrix(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    cat_idx: Optional[np.ndarray] = None,
    num_idx: Optional[np.ndarray] = None,
    feature_ranges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Gower distance between each row of A and each row of B.
    Assumes A and B have same number/order of features.
    """
    n_features = matrix_a.shape[1]
    if cat_idx is None and num_idx is None:
        num_mask = np.ones(n_features, dtype=bool)
        cat_mask = np.zeros(n_features, dtype=bool)
    else:
        if isinstance(cat_idx, (list, tuple, np.ndarray)) and not isinstance(cat_idx, np.ndarray):
            cat_idx = np.array(cat_idx)
        if isinstance(num_idx, (list, tuple, np.ndarray)) and not isinstance(num_idx, np.ndarray):
            num_idx = np.array(num_idx)
        cat_mask = np.zeros(n_features, dtype=bool)
        num_mask = np.zeros(n_features, dtype=bool)
        if cat_idx is not None:
            cat_mask[cat_idx] = True if cat_idx.dtype != bool else cat_idx
        if num_idx is not None:
            num_mask[num_idx] = True if num_idx.dtype != bool else num_idx
        if not (cat_mask.any() or num_mask.any()):
            num_mask[:] = True

    # Numeric part
    dist_num = 0.0
    if num_mask.any():
        a_num = matrix_a[:, num_mask].astype(float)
        b_num = matrix_b[:, num_mask].astype(float)
        if feature_ranges is None:
            ab = np.vstack([a_num, b_num])
            ranges = np.ptp(ab, axis=0)
        else:
            ranges = feature_ranges
            if ranges.shape[0] != a_num.shape[1]:
                raise ValueError("feature_ranges length must match number of numeric features")
        ranges = np.where(ranges == 0, 1.0, ranges)
        dist_num = np.abs(a_num[:, None, :] - b_num[None, :, :]) / ranges[None, None, :]

    # Categorical part
    if cat_mask.any():
        a_cat = matrix_a[:, cat_mask]
        b_cat = matrix_b[:, cat_mask]
        eq = (a_cat[:, None, :] == b_cat[None, :, :]).astype(float)
        dist_cat = 1.0 - eq
        if isinstance(dist_num, float):
            dist = dist_cat
        else:
            dist = np.concatenate([dist_num, dist_cat], axis=2)
    else:
        dist = dist_num

    if isinstance(dist, float):
        return np.zeros((matrix_a.shape[0], matrix_b.shape[0]), dtype=float)
    return dist.mean(axis=2)


def gower_distance(vec_a: np.ndarray, vec_b: np.ndarray, feature_ranges: Optional[np.ndarray] = None) -> float:
    """Gower distance between 1D vectors in [0,1]. Numeric-only variant."""
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    if feature_ranges is None:
        ranges = np.ptp(np.vstack([vec_a, vec_b]), axis=0)
    else:
        ranges = feature_ranges
        if ranges.shape[0] != vec_a.shape[0]:
            raise ValueError("feature_ranges length must match vector length")
    ranges = np.where(ranges == 0, 1.0, ranges)
    dist = np.abs(vec_a - vec_b) / ranges
    return float(np.mean(dist))
