from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from case_align.metrics import (
    cosine_similarity,
    gower_distance,
    gower_distance_matrix,
    rankdata,
    safe_normalise_rows,
    sim_to_dist,
    spearman_similarity,
)
from case_align.neighborhood import (
    nearest_unlikes,
    neighbours_like,
    neighbours_like_of_anchor,
)


@dataclass
class RobustnessResult:
    index: int
    S_plus: float
    S_minus: float
    R_ratio: float
    R_bounded: float
    nun_indices: List[int]
    nun_distances: List[float]
    k_like_x: int
    k_like_u: int
    S_minus_u: float = 0.0
    S_minus_x: float = 0.0


class RobustnessCBR:
    """
    Case Alignment for robustness.

    - Problem space uses `problem_metric`
    - Solution space uses `sim_metric`

    IMPORTANT: For these experiments, we require both metrics to match.
    """

    def __init__(
        self,
        k: int = 5,
        m_unlike: int = 1,
        sim_metric: str = "gower",            # "gower", "cosine", or "spearman"
        problem_metric: str = "gower",        # "gower", "cosine", or "spearman"
        cat_idx: Optional[Union[np.ndarray, List[int]]] = None,
        num_idx: Optional[Union[np.ndarray, List[int]]] = None,
        feature_ranges: Optional[np.ndarray] = None,
        expl_feature_ranges: Optional[np.ndarray] = None,
        weight_sigma: Optional[float] = None, # if set, use exp(-d/sigma) neighbour weights
        epsilon: float = 1e-8,
        random_state: Optional[int] = None,
        like_only: bool = False,
        robust_mode: str = "geom",            # "ratio" (legacy) or "geom" (geom mean of S+, S-_u, S-_x)
    ):
        if problem_metric != sim_metric:
            raise ValueError("problem_metric and sim_metric must match for these experiments")

        self.k = k
        self.m_unlike = m_unlike
        self.sim_metric = sim_metric
        self.problem_metric = problem_metric
        self.cat_idx = np.array(cat_idx) if cat_idx is not None else None
        self.num_idx = np.array(num_idx) if num_idx is not None else None
        self.feature_ranges = feature_ranges
        self.expl_feature_ranges = expl_feature_ranges
        self.weight_sigma = weight_sigma
        self.epsilon = epsilon
        self.random_state = random_state
        self.like_only = like_only
        self.robust_mode = robust_mode

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.expl: Optional[np.ndarray] = None

        self._dist_mat: Optional[np.ndarray] = None
        self._feature_ranges_used: Optional[np.ndarray] = None
        self._expl_feature_ranges_used: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    # --- Fit and caching ---
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        explanations: np.ndarray,
    ) -> "RobustnessCBR":
        """Fit with data and precomputed explanations aligned to X."""
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.expl = np.asarray(explanations)

        if self.expl.shape[0] != self.X.shape[0]:
            raise ValueError("Explanations must align with X rows.")

        self._cache_expl_feature_ranges()
        self._cache_problem_feature_ranges()
        self._precompute_problem_distance_matrix()
        return self

    def _cache_expl_feature_ranges(self) -> None:
        """Cache feature ranges for explanation (solution) space if needed."""
        if self.sim_metric != "gower":
            return
        self._expl_feature_ranges_used = self.expl_feature_ranges
        if self._expl_feature_ranges_used is None:
            ranges = np.ptp(self.expl.astype(float), axis=0)
            self._expl_feature_ranges_used = np.where(ranges == 0, 1.0, ranges)
        elif self._expl_feature_ranges_used.shape[0] != self.expl.shape[1]:
            raise ValueError("expl_feature_ranges length must match number of explanation features")

    def _cache_problem_feature_ranges(self) -> None:
        """Cache feature ranges for problem space (numeric columns)."""
        self._feature_ranges_used = self.feature_ranges
        if self._feature_ranges_used is not None:
            return

        n_features = self.X.shape[1]
        if self.cat_idx is None and self.num_idx is None:
            num_mask = np.ones(n_features, dtype=bool)
        else:
            cat_mask = np.zeros(n_features, dtype=bool)
            num_mask = np.zeros(n_features, dtype=bool)
            if self.cat_idx is not None:
                cidx = self.cat_idx if isinstance(self.cat_idx, np.ndarray) else np.array(self.cat_idx)
                cat_mask[cidx] = True if cidx.dtype != bool else cidx
            if self.num_idx is not None:
                nidx = self.num_idx if isinstance(self.num_idx, np.ndarray) else np.array(self.num_idx)
                num_mask[nidx] = True if nidx.dtype != bool else nidx
            if not (cat_mask.any() or num_mask.any()):
                num_mask[:] = True

        if num_mask.any():
            ranges = np.ptp(self.X[:, num_mask].astype(float), axis=0)
            self._feature_ranges_used = np.where(ranges == 0, 1.0, ranges)
        else:
            self._feature_ranges_used = None

    def _precompute_problem_distance_matrix(self) -> None:
        """Precompute pairwise distances in problem space."""
        if self.problem_metric == "gower":
            self._dist_mat = gower_distance_matrix(
                self.X,
                self.X,
                cat_idx=self.cat_idx,
                num_idx=self.num_idx,
                feature_ranges=self._feature_ranges_used,
            )
        elif self.problem_metric == "cosine":
            normalised = safe_normalise_rows(self.X.astype(float))
            similarity = normalised @ normalised.T
            self._dist_mat = 1.0 - 0.5 * (similarity + 1.0)
        elif self.problem_metric == "spearman":
            ranks = np.vstack([rankdata(row) for row in self.X.astype(float)])
            ranks = ranks - ranks.mean(axis=1, keepdims=True)
            ranks = safe_normalise_rows(ranks)
            similarity = ranks @ ranks.T
            self._dist_mat = 1.0 - 0.5 * (similarity + 1.0)
        else:
            raise ValueError(f"Unknown problem_metric: {self.problem_metric}")

    # --- Neighborhood helpers (problem space) ---
    def _problem_dists_to(self, index: int) -> np.ndarray:
        if self._dist_mat is None:
            raise RuntimeError("Distance matrix is not initialized.")
        return self._dist_mat[index]

    def _neighbours_like(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        dists = self._problem_dists_to(index)
        return neighbours_like(self.y, dists, index, self.k)

    def _nearest_unlikes(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        dists = self._problem_dists_to(index)
        return nearest_unlikes(self.y, dists, index, self.m_unlike)

    def _neighbours_like_of_anchor(self, anchor_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        dists = self._problem_dists_to(anchor_idx)
        return neighbours_like_of_anchor(self.y, dists, anchor_idx, self.k)

    # --- Weights and distances ---
    def _weight_from_dists(self, dists: np.ndarray) -> np.ndarray:
        if self.weight_sigma is None:
            return np.ones_like(dists, dtype=float)
        weights = np.exp(-dists / max(self.weight_sigma, self.epsilon))
        total = weights.sum()
        return weights / (total if total > 0 else 1.0)

    def _solution_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Distance in solution space using sim_metric."""
        if self.sim_metric == "gower":
            return gower_distance(vec_a, vec_b, feature_ranges=self._expl_feature_ranges_used)
        if self.sim_metric == "cosine":
            sim = cosine_similarity(vec_a, vec_b, eps=self.epsilon)
        elif self.sim_metric == "spearman":
            sim = spearman_similarity(vec_a, vec_b, eps=self.epsilon)
        else:
            raise ValueError(f"Unknown sim_metric: {self.sim_metric}")
        return sim_to_dist(sim)

    # --- Case alignment core ---
    def _align_scores(self, index: int) -> np.ndarray:
        """Return distances in solution space from index to all cases."""
        return np.array([self._solution_distance(self.expl[index], self.expl[j]) for j in range(self.expl.shape[0])])

    def _case_alignment(self, index: int, neighbour_indices: np.ndarray) -> float:
        """
        CaseAlign(t) = sum_i (1 - Dprob(t, c_i)) * Align(t, c_i)
                       / sum_i (1 - Dprob(t, c_i))
        """
        if neighbour_indices.size == 0:
            return 0.0

        dprob = self._problem_dists_to(index)[neighbour_indices]
        dsoln_all = self._align_scores(index)
        dsoln_neigh = dsoln_all[neighbour_indices]

        align = self._alignment_scores(dsoln_all, dsoln_neigh)
        return self._weighted_alignment(dprob, align)

    def _alignment_scores(self, dsoln_all: np.ndarray, dsoln_neigh: np.ndarray) -> np.ndarray:
        """
        Align(t, c_i) = 1 - (Dsoln(t, c_i) - min Dsoln) / (max Dsoln - min Dsoln)
        """
        ds_min = float(np.min(dsoln_all))
        ds_max = float(np.max(dsoln_all))
        denom = max(ds_max - ds_min, self.epsilon)
        return 1.0 - (dsoln_neigh - ds_min) / denom

    def _weighted_alignment(self, dprob: np.ndarray, align: np.ndarray) -> float:
        """Weighted average using (1 - Dprob) weights."""
        weights = 1.0 - dprob
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return 0.0
        return float(np.sum(weights * align) / weight_sum)

    # --- Public API ---
    def compute_for_index(self, index: int) -> RobustnessResult:
        if self.X is None or self.expl is None:
            raise RuntimeError("Call fit() first.")

        like_indices_x, _ = self._neighbours_like(index)
        S_plus = self._case_alignment(index, like_indices_x)

        nun_indices, nun_dists = self._nearest_unlikes(index)
        if self.like_only or nun_indices.size == 0:
            S_minus = 0.0
            S_minus_u = 0.0
            S_minus_x = 0.0
            nun_list = []
            nun_distances = []
        else:
            S_minus_u_list = [self._case_alignment(a_idx, self._neighbours_like_of_anchor(a_idx)[0]) for a_idx in nun_indices]
            S_minus_x_list = [self._case_alignment(index, self._neighbours_like_of_anchor(a_idx)[0]) for a_idx in nun_indices]
            anchor_weights = self._weight_from_dists(nun_dists)
            S_minus_u = float(np.sum(anchor_weights * np.array(S_minus_u_list)))
            S_minus_x = float(np.sum(anchor_weights * np.array(S_minus_x_list)))
            S_minus = 0.5 * (S_minus_u + S_minus_x)
            nun_list = nun_indices.tolist()
            nun_distances = nun_dists.tolist()

        if self.like_only:
            R_ratio = np.inf if S_plus > 0 else 1.0
            R_bounded = S_plus
        else:
            if self.robust_mode == "ratio":
                R_ratio = (S_plus / (S_minus + self.epsilon)) if S_minus > 0 else np.inf if S_plus > 0 else 1.0
                R_bounded = S_plus / (S_plus + S_minus + self.epsilon)
            elif self.robust_mode == "geom":
                R_ratio = (S_plus / (S_minus + self.epsilon)) if S_minus > 0 else np.inf if S_plus > 0 else 1.0
                a = max(S_plus, self.epsilon)
                b = max(S_minus_u, self.epsilon)
                c = 1 - max(S_minus_x, self.epsilon)
                R_bounded = float((a * b * c) ** (1.0 / 3.0))
            else:
                raise ValueError("robust_mode must be 'ratio' or 'geom'")

        return RobustnessResult(
            index=index,
            S_plus=S_plus,
            S_minus=S_minus,
            S_minus_u=S_minus_u,
            S_minus_x=S_minus_x,
            R_ratio=R_ratio,
            R_bounded=R_bounded,
            nun_indices=nun_list,
            nun_distances=nun_distances,
            k_like_x=int(like_indices_x.size),
            k_like_u=int(self.k),
        )

    def compute_all(self) -> List[RobustnessResult]:
        if self.X is None:
            raise RuntimeError("Call fit() first.")
        return [self.compute_for_index(i) for i in range(self.X.shape[0])]
