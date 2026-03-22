import unittest

import torch

from romav2.device import device
from romav2.romav2 import RoMaV2, kde


def _normalized_identity_warp(h: int, w: int) -> torch.Tensor:
    y = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
    x = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


class _SampleHarness:
    bidirectional = False


class SamplingRobustnessTests(unittest.TestCase):
    def test_kde_large_tensor_is_finite(self) -> None:
        x = torch.rand((20_000, 4), device=device)
        density = kde(x, std=0.1, max_reference_points=4096, chunk_size=1024)
        self.assertEqual(tuple(density.shape), (20_000,))
        self.assertTrue(bool(torch.isfinite(density).all().item()))

    def test_sample_handles_zero_confidence(self) -> None:
        h, w = 24, 24
        warp = _normalized_identity_warp(h, w)
        overlap = torch.zeros((1, h, w, 1), device=device)
        precision = torch.zeros((1, h, w, 2, 2), device=device)
        preds = {
            "warp_AB": warp,
            "overlap_AB": overlap,
            "precision_AB": precision,
        }

        sampled = RoMaV2.sample(_SampleHarness(), preds, num_corresp=128)
        matches, confidence, precision_ab, precision_ba = sampled

        self.assertGreater(matches.shape[0], 0)
        self.assertEqual(matches.shape[-1], 4)
        self.assertEqual(confidence.ndim, 1)
        self.assertIsNotNone(precision_ab)
        self.assertIsNotNone(precision_ba)

    def test_sample_handles_missing_precision(self) -> None:
        h, w = 16, 16
        warp = _normalized_identity_warp(h, w)
        overlap = torch.full((1, h, w, 1), 0.5, device=device)
        preds = {
            "warp_AB": warp,
            "overlap_AB": overlap,
        }

        matches, confidence, precision_ab, precision_ba = RoMaV2.sample(
            _SampleHarness(), preds, num_corresp=64
        )
        self.assertGreater(matches.shape[0], 0)
        self.assertEqual(confidence.ndim, 1)
        self.assertIsNone(precision_ab)
        self.assertIsNone(precision_ba)


if __name__ == "__main__":
    unittest.main()
