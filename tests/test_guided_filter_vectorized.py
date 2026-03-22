import unittest

import numpy as np

from run_romav2_pair import _guided_filter, _guided_filter_with_mp_limit


class GuidedFilterVectorizedTests(unittest.TestCase):
    def test_multichannel_matches_per_channel(self) -> None:
        rng = np.random.default_rng(0)
        guide = rng.random((64, 96), dtype=np.float32)
        src = rng.random((64, 96, 2), dtype=np.float32)
        radius = 5
        eps = 1e-3

        multi = _guided_filter(guide, src, radius, eps)
        per_channel = np.stack(
            [_guided_filter(guide, src[..., i], radius, eps) for i in range(src.shape[-1])],
            axis=-1,
        )
        self.assertEqual(multi.shape, src.shape)
        self.assertTrue(np.allclose(multi, per_channel, atol=1e-5, rtol=1e-5))

    def test_mp_limited_filter_preserves_shape(self) -> None:
        rng = np.random.default_rng(1)
        guide = rng.random((256, 320), dtype=np.float32)
        src = rng.random((256, 320, 2), dtype=np.float32)

        out = _guided_filter_with_mp_limit(
            guide=guide,
            src=src,
            radius=9,
            eps=1e-3,
            max_megapixels=0.08,
        )
        self.assertEqual(out.shape, src.shape)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
