import unittest

import numpy as np

from case_align.case_align import RobustnessCBR


def align_local(cbr: RobustnessCBR, i: int, neigh_idx: np.ndarray, eps: float = 1e-8) -> float:
    if neigh_idx.size == 0:
        return 0.0
    dsoln_neigh = np.array([cbr._solution_distance(cbr.expl[i], cbr.expl[j]) for j in neigh_idx])  # noqa: SLF001
    dsmin = float(np.min(dsoln_neigh))
    dsmax = float(np.max(dsoln_neigh))
    denom = max(dsmax - dsmin, eps)
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    return float(np.mean(align))


def farthest_unlike_anchor(cbr: RobustnessCBR, i: int, y: np.ndarray):
    unlike_idx = np.where(y != y[i])[0]
    if unlike_idx.size == 0:
        return None
    dprob = cbr._problem_dists_to(i)  # noqa: SLF001
    return int(unlike_idx[np.argmax(dprob[unlike_idx])])


def geom(S_plus: float, S_minus_u: float, S_minus_x: float, eps: float = 1e-8) -> float:
    a = max(S_plus, eps)
    b = max(S_minus_u, eps)
    c = 1.0 - max(S_minus_x, eps)
    return float((a * b * c) ** (1.0 / 3.0))


class TestTightnessIdenticalScenario(unittest.TestCase):
    def test_identical_scenario_expected_behavior(self):
        # Simple 1D data: class 0 near 0, class 1 far away
        X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        y = np.array([0, 0, 0, 1, 1, 1])
        expl = X.copy()

        cbr = RobustnessCBR(
            k=2,
            m_unlike=1,
            sim_metric="gower",
            problem_metric="gower",
            robust_mode="geom",
        ).fit(X, y, explanations=expl)

        i = 0  # query in class 0
        far_anchor = farthest_unlike_anchor(cbr, i, y)
        self.assertIsNotNone(far_anchor)

        # Identical neighbors for x (S_plus)
        neigh_x = np.array([i, i])
        S_plus = align_local(cbr, i, neigh_x)

        # Identical neighbors for farthest unlike anchor u (S_minus_u)
        u = far_anchor
        neigh_u = np.array([u, u])
        S_minus_u = align_local(cbr, u, neigh_u)

        # x vs u-like (identical to u) for S_minus_x
        S_minus_x = align_local(cbr, i, neigh_u)

        R_bounded = geom(S_plus, S_minus_u, S_minus_x)

        # Expectations
        self.assertAlmostEqual(S_plus, 1.0, places=6)
        self.assertAlmostEqual(S_minus_u, 1.0, places=6)
        # With local normalization and identical neighbors, dsmin == dsmax,
        # so alignment collapses to 1 even if x is far from u.
        self.assertAlmostEqual(S_minus_x, 1.0, places=6)
        # Therefore R_bounded collapses to 0 due to (1 - S_minus_x).
        self.assertLess(R_bounded, 0.2)


if __name__ == "__main__":
    unittest.main()
