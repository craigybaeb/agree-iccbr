import unittest

import numpy as np

from explainers.captum_explain import explain_batch, TORCH_AVAILABLE


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is required for captum tests.")
class TestCaptumExplain(unittest.TestCase):
    def test_explain_batch_shapes(self):
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        model.eval()

        X = np.random.randn(10, 4).astype(np.float32)
        attrs = explain_batch(model, X, methods=("lrp", "dl", "ig"), batch_size=5)

        self.assertIn("lrp", attrs)
        self.assertIn("dl", attrs)
        self.assertIn("ig", attrs)

        for name, a in attrs.items():
            self.assertEqual(a.shape, torch.Size([10, 4]), f"{name} shape mismatch")


if __name__ == "__main__":
    unittest.main()
