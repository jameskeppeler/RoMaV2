import unittest

import torch
from PIL import Image

from run_romav2_pair import (
    _alignment_score_from_warp,
    _map_scalar_through_warp,
    _pil_to_tensor,
    _rgb_chw_to_luma,
    _warp_compose,
)


def _identity_warp(h: int, w: int) -> torch.Tensor:
    y = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, dtype=torch.float32)
    x = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


class IterativeRematchMathTests(unittest.TestCase):
    def test_warp_compose_identity(self) -> None:
        h, w = 24, 32
        base = _identity_warp(h, w)
        delta = _identity_warp(h, w)
        out = _warp_compose(warp_base=base, warp_delta=delta)
        self.assertTrue(torch.allclose(out, base, atol=1e-5, rtol=1e-5))

    def test_warp_compose_identity_base_equals_delta(self) -> None:
        h, w = 24, 32
        base = _identity_warp(h, w)
        delta = _identity_warp(h, w).clone()
        delta[..., 0] = (delta[..., 0] + 0.04).clamp(-1.0, 1.0)
        out = _warp_compose(warp_base=base, warp_delta=delta)
        # Ignore outer border where grid_sample border padding can clip shifts.
        core_out = out[1:-1, 1:-1]
        core_delta = delta[1:-1, 1:-1]
        self.assertTrue(torch.allclose(core_out, core_delta, atol=2e-3, rtol=2e-3))

    def test_scalar_map_through_identity(self) -> None:
        h, w = 20, 28
        scalar = torch.linspace(0.0, 1.0, h * w, dtype=torch.float32).reshape(h, w)
        warp = _identity_warp(h, w)
        sampled = _map_scalar_through_warp(scalar_map=scalar, warp=warp)
        self.assertTrue(torch.allclose(sampled, scalar, atol=1e-5, rtol=1e-5))

    def test_alignment_score_prefers_correct_shift(self) -> None:
        h, w = 64, 64
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, dtype=torch.float32),
            indexing="ij",
        )
        base = (0.5 * torch.sin(12.0 * xx) + 0.5 * torch.cos(10.0 * yy)).clamp(-1.0, 1.0)
        base = ((base + 1.0) * 0.5).clamp(0.0, 1.0)
        ref_rgb = torch.stack((base, base, base), dim=-1)
        ref_u8 = (ref_rgb.numpy() * 255.0).round().astype("uint8")
        ref_pil = Image.fromarray(ref_u8, mode="RGB")

        shift_px = 4
        src = torch.zeros_like(base)
        src[:, shift_px:] = base[:, :-shift_px]
        src[:, :shift_px] = base[:, :1]
        src_rgb = torch.stack((src, src, src), dim=-1)
        src_u8 = (src_rgb.numpy() * 255.0).round().astype("uint8")
        src_pil = Image.fromarray(src_u8, mode="RGB")

        device = torch.device("cpu")
        src_tensor = _pil_to_tensor(src_pil, device=device)
        ref_luma = _rgb_chw_to_luma(_pil_to_tensor(ref_pil, device=device)[0])

        warp_identity = _identity_warp(20, 20)
        dx = (2.0 * float(shift_px)) / float(w)
        warp_shift = warp_identity.clone()
        warp_shift[..., 0] = (warp_shift[..., 0] + dx).clamp(-1.0, 1.0)

        score_identity = _alignment_score_from_warp(
            ref_luma_full=ref_luma,
            src_tensor_full=src_tensor,
            warp_small=warp_identity,
            overlap_small=None,
            out_h=h,
            out_w=w,
        )["score"]
        score_shift = _alignment_score_from_warp(
            ref_luma_full=ref_luma,
            src_tensor_full=src_tensor,
            warp_small=warp_shift,
            overlap_small=None,
            out_h=h,
            out_w=w,
        )["score"]
        self.assertLess(score_shift, score_identity)


if __name__ == "__main__":
    unittest.main()
