import unittest

import numpy as np

from operator_data import eon_pkl_to_grid, mat_to_points


class TestOperatorData(unittest.TestCase):
    def test_mat_to_points_shapes(self):
        bsz = 5
        s = 64
        mat = {
            "a": np.random.randn(bsz, s).astype(np.float32),
            "u": np.random.randn(bsz, s).astype(np.float32),
        }
        xq, us, yq, meta = mat_to_points(
            mat,
            "burgers",
            num_sensors=16,
            num_query=20,
            sensor_strategy="uniform_idx",
            query_strategy="random",
            seed=1,
        )
        self.assertEqual(tuple(xq.shape), (20, 1))
        self.assertEqual(tuple(us.shape), (bsz, 16))
        self.assertEqual(tuple(yq.shape), (bsz, 20, 1))
        self.assertEqual(len(meta["sensor_indices"]), 16)

    def test_eon_pkl_to_grid_meta_required(self):
        t = np.random.rand(4, 1).astype(np.float32)
        y = np.random.rand(4, 32).astype(np.float32)
        u = np.random.rand(4, 8).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "meta\['x_grid'\]"):
            eon_pkl_to_grid((t, y, u), meta={"sensor_locs": np.linspace(0, 1, 8)})


if __name__ == "__main__":
    unittest.main()
