import unittest

import numpy as np


def compute_align(dsoln_all: np.ndarray, neigh_idx: np.ndarray, eps: float = 1e-8):
    dsmin = float(np.min(dsoln_all))
    dsmax = float(np.max(dsoln_all))
    denom = max(dsmax - dsmin, eps)
    dsoln_neigh = dsoln_all[neigh_idx]
    align = 1.0 - (dsoln_neigh - dsmin) / denom
    return align


def assert_in_01(arr):
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise AssertionError(f"out of bounds: min={arr.min()}, max={arr.max()}")


class TestAlignBoundedness(unittest.TestCase):
    def test_normal_mixed(self):
        d = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
        idx = np.array([0, 2, 4])
        align = compute_align(d, idx)
        assert_in_01(align)

    def test_all_equal(self):
        d = np.array([5.0, 5.0, 5.0, 5.0])
        idx = np.array([0, 1, 2, 3])
        align = compute_align(d, idx)
        assert_in_01(align)
        self.assertTrue(np.allclose(align, 1.0))

    def test_negative_only(self):
        d = np.array([-10.0, -5.0, -1.0])
        idx = np.array([0, 1, 2])
        align = compute_align(d, idx)
        assert_in_01(align)

    def test_positive_only(self):
        d = np.array([0.5, 2.0, 10.0])
        idx = np.array([0, 2])
        align = compute_align(d, idx)
        assert_in_01(align)

    def test_large_range(self):
        d = np.array([-1e6, 0.0, 1e6])
        idx = np.array([0, 1, 2])
        align = compute_align(d, idx)
        assert_in_01(align)

    def test_single_neighbor(self):
        d = np.array([-3.0, 4.0, 9.0])
        idx = np.array([1])
        align = compute_align(d, idx)
        assert_in_01(align)

    def test_random_mixture(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=100)
        idx = rng.choice(100, size=10, replace=False)
        align = compute_align(d, idx)
        assert_in_01(align)


if __name__ == "__main__":
    unittest.main()
