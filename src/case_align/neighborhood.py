from typing import Tuple

import numpy as np


def neighbours_like(labels: np.ndarray, distances: np.ndarray, index: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and distances of like-class neighbours of index (excluding itself)."""
    label = labels[index]
    mask_like = (labels == label)
    mask_like[index] = False
    like_idx = np.where(mask_like)[0]
    like_dists = distances[like_idx]
    order = np.argsort(like_dists)
    sel = order[: min(k, like_idx.size)]
    return like_idx[sel], like_dists[sel]


def nearest_unlikes(labels: np.ndarray, distances: np.ndarray, index: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and distances of the m nearest unlike neighbours of index."""
    label = labels[index]
    mask_unlike = (labels != label)
    unlike_idx = np.where(mask_unlike)[0]
    unlike_dists = distances[unlike_idx]
    order = np.argsort(unlike_dists)
    m = min(m, unlike_idx.size)
    sel = order[:m]
    return unlike_idx[sel], unlike_dists[sel]


def neighbours_like_of_anchor(labels: np.ndarray, distances: np.ndarray, anchor_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Like-class neighbours around a given anchor (exclude the anchor)."""
    label = labels[anchor_idx]
    mask_like = (labels == label)
    mask_like[anchor_idx] = False
    like_idx = np.where(mask_like)[0]
    like_dists = distances[like_idx]
    order = np.argsort(like_dists)
    sel = order[: min(k, like_idx.size)]
    return like_idx[sel], like_dists[sel]
