import unittest

import numpy as np
import torch

from run_romav2_pair import _pred_overlap_mean, _trim_correspondences_for_prealign


class PrealignFilterTests(unittest.TestCase):
    def test_pred_overlap_mean(self) -> None:
        overlap = torch.full((1, 8, 8, 1), 0.25, dtype=torch.float32)
        preds = {"overlap_AB": overlap}
        mean = _pred_overlap_mean(preds)
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(float(mean), 0.25, places=6)
        self.assertIsNone(_pred_overlap_mean({}))

    def test_trim_correspondences_filters_borders(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.uniform(low=0.0, high=1000.0, size=(2000, 2)).astype(np.float32)
        ref = src + rng.normal(0.0, 2.0, size=(2000, 2)).astype(np.float32)

        src_trim, ref_trim, applied = _trim_correspondences_for_prealign(
            src_pts=src,
            ref_pts=ref,
            trim_frac=0.10,
            min_points=80,
        )
        self.assertTrue(applied)
        self.assertLess(src_trim.shape[0], src.shape[0])
        self.assertEqual(src_trim.shape[0], ref_trim.shape[0])
        self.assertGreaterEqual(src_trim.shape[0], 80)


if __name__ == "__main__":
    unittest.main()
