import os
import tempfile
import unittest

import torch

from extremonet.io import load_eon, save_eon
from extremonet.model import ExtremeLearning, ExtremONet


class TestExtremONetIO(unittest.TestCase):
    def test_save_load_consistency(self):
        trunk = ExtremeLearning(indim=1, outdim=64, c=1, s=5.0, seed=0)
        branch = ExtremeLearning(indim=10, outdim=64, c=2, s=0.1, seed=1)
        model = ExtremONet(trunk=trunk, branch=branch, outdim=1)

        with torch.no_grad():
            model.A.copy_(torch.randn_like(model.A))
            model.B.copy_(torch.randn_like(model.B))

        xq = torch.linspace(0, 1, 20).reshape(-1, 1)
        us = torch.randn(3, 10)
        yp_ref = model.predict_tensor(xq, us)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "eon.pt")
            save_eon(model, path, extra_meta={"foo": "bar"})
            loaded, meta = load_eon(path)
            yp_loaded = loaded.predict_tensor(xq, us)

        self.assertIn("foo", meta)
        self.assertTrue(torch.allclose(yp_ref, yp_loaded, atol=1e-6))
        self.assertIn("trunk.R", loaded.state_dict())
        self.assertIn("trunk.b", loaded.state_dict())
        self.assertIn("branch.R", loaded.state_dict())
        self.assertIn("branch.b", loaded.state_dict())


if __name__ == "__main__":
    unittest.main()
