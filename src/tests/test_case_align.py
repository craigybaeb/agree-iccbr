import unittest

import numpy as np

from case_align.case_align import RobustnessCBR


class TestCaseAlign(unittest.TestCase):
    def setUp(self):
        # Simple synthetic dataset with two classes
        self.X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.2, 1.0],
            ],
            dtype=float,
        )
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.expl = self.X.copy()

    def test_metric_mismatch_raises(self):
        with self.assertRaises(ValueError):
            RobustnessCBR(sim_metric="gower", problem_metric="cosine")

    def test_case_alignment_in_range(self):
        cbr = RobustnessCBR(k=2, m_unlike=1, sim_metric="gower", problem_metric="gower").fit(
            self.X, self.y, explanations=self.expl
        )
        for i in range(self.X.shape[0]):
            result = cbr.compute_for_index(i)
            self.assertGreaterEqual(result.S_plus, 0.0)
            self.assertLessEqual(result.S_plus, 1.0)
            self.assertGreaterEqual(result.S_minus_u, 0.0)
            self.assertLessEqual(result.S_minus_u, 1.0)
            self.assertGreaterEqual(result.S_minus_x, 0.0)
            self.assertLessEqual(result.S_minus_x, 1.0)

    def test_compute_all_length(self):
        cbr = RobustnessCBR(k=2, m_unlike=1, sim_metric="gower", problem_metric="gower").fit(
            self.X, self.y, explanations=self.expl
        )
        results = cbr.compute_all()
        self.assertEqual(len(results), self.X.shape[0])

    def test_alignment_scores_shape(self):
        cbr = RobustnessCBR(k=2, m_unlike=1, sim_metric="gower", problem_metric="gower").fit(
            self.X, self.y, explanations=self.expl
        )
        dsoln_all = cbr._align_scores(0)
        neigh_idx, _ = cbr._neighbours_like(0)
        dsoln_neigh = dsoln_all[neigh_idx]
        align = cbr._alignment_scores(dsoln_all, dsoln_neigh)
        self.assertEqual(align.shape[0], dsoln_neigh.shape[0])


if __name__ == "__main__":
    unittest.main()
