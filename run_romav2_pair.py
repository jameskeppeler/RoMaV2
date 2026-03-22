from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RoMa v2 dense matching on a reference/source pair and save diagnostics."
    )
    parser.add_argument(
        "--ref",
        required=True,
        type=Path,
        help="Path to reference/original image (target frame).",
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Path to source/deviated image (image to warp toward --ref).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory for diagnostics.",
    )
    parser.add_argument(
        "--setting",
        default="precise",
        choices=["turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast"],
        help="RoMa v2 setting preset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Requested number of sampled correspondences.",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=1200,
        help="Maximum sampled correspondences to draw in the correspondence visualization.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile during model construction (off by default for robustness).",
    )
    parser.add_argument(
        "--regularize-overlap-thresh",
        type=float,
        default=0.35,
        help=(
            "Overlap threshold used to compute confidence blend alpha for regularized warp. "
            "Higher values trust dense warp less in uncertain regions."
        ),
    )
    parser.add_argument(
        "--regularize-fallback",
        choices=["none", "identity", "reference"],
        default="identity",
        help=(
            "Fallback image used where overlap is low for regularized warp: "
            "'identity' samples source with identity grid, 'reference' uses reference image, "
            "'none' disables regularized output."
        ),
    )
    parser.add_argument(
        "--guided-filter-radius",
        type=int,
        default=16,
        help=(
            "Radius for edge-preserving guided filter applied to the warp field. "
            "Uses the reference (B&W original) as the edge guide so the warp snaps "
            "to original geometry. 0 disables guided filtering."
        ),
    )
    parser.add_argument(
        "--guided-filter-eps",
        type=float,
        default=1e-3,
        help=(
            "Regularization (epsilon) for guided filter. Smaller values preserve "
            "edges more aggressively; larger values produce smoother warps."
        ),
    )
    parser.add_argument(
        "--chroma-filter-radius",
        type=int,
        default=48,
        help=(
            "Radius for guided filter applied to chrominance (a/b) channels "
            "before color blending. Larger than warp radius because color "
            "varies slowly and the AI colorizer introduces geometry shifts of "
            "10-30px that need smoothing over.  0 disables chrominance filtering."
        ),
    )
    parser.add_argument(
        "--color-opacity",
        type=float,
        default=1.0,
        help=(
            "Opacity for color transfer in [0, 1]. "
            "1.0 = full transferred color, 0.0 = keep base image unchanged."
        ),
    )
    parser.add_argument(
        "--no-color-transfer",
        action="store_true",
        help=(
            "Disable color transfer step. By default the pipeline applies a "
            "Photoshop-style Color blend (base luminance + warped source color)."
        ),
    )
    parser.add_argument(
        "--save-dense-preds",
        action="store_true",
        help=(
            "Save every dense prediction tensor as .npy files. Disabled by default "
            "because this can be slow and large on high-resolution runs."
        ),
    )
    parser.add_argument(
        "--diag-max-side",
        type=int,
        default=2048,
        help=(
            "Maximum side length for diagnostic visualizations (triptych, difference, "
            "correspondences). Lower values are much faster on large images."
        ),
    )
    parser.add_argument(
        "--filter-max-megapixels",
        type=float,
        default=12.0,
        help=(
            "Maximum megapixels used by guided-filter operations. If images are larger, "
            "guided filtering is performed on a downscaled copy and upsampled back."
        ),
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help=(
            "Run a minimal warmup pass (model init + one dense match) and exit. "
            "Used to pre-download model weights/caches before a real run."
        ),
    )
    parser.add_argument(
        "--disable-auto-border-crop",
        action="store_true",
        help=(
            "Disable automatic border/frame crop detection on ref/src before matching. "
            "Enabled by default to reduce frame-card alignment failures."
        ),
    )
    parser.add_argument(
        "--border-crop-max-frac",
        type=float,
        default=0.22,
        help="Maximum fraction removable from each side during auto border crop.",
    )
    parser.add_argument(
        "--disable-global-prealign",
        action="store_true",
        help=(
            "Disable global similarity pre-alignment (scale/rotation/translation) "
            "before the final dense match."
        ),
    )
    parser.add_argument(
        "--min-global-inlier-ratio",
        type=float,
        default=0.20,
        help="Minimum RANSAC inlier ratio required to apply global pre-alignment.",
    )
    parser.add_argument(
        "--min-overlap-mean",
        type=float,
        default=0.10,
        help=(
            "Low-overlap guardrail threshold. If final overlap mean is below this value "
            "the run aborts unless --allow-low-overlap is set."
        ),
    )
    parser.add_argument(
        "--allow-low-overlap",
        action="store_true",
        help="Allow continuing even if overlap mean is below --min-overlap-mean.",
    )
    parser.add_argument(
        "--auto-reference-fallback-overlap",
        type=float,
        default=0.20,
        help=(
            "If final overlap mean is below this value and requested fallback is "
            "'identity', auto-switch regularization fallback to 'reference'."
        ),
    )
    return parser.parse_args()


def _box_filter_2d(src: np.ndarray, radius: int) -> np.ndarray:
    """Fast O(N) box filter using cumulative sums (separable, two-pass)."""
    ksize = 2 * radius + 1

    # Horizontal pass via cumsum
    padded = np.pad(src, ((0, 0), (radius + 1, radius)), mode="reflect")
    cs = np.cumsum(padded, axis=1)
    horiz = (cs[:, ksize:] - cs[:, :-ksize]) / ksize

    # Vertical pass via cumsum
    padded = np.pad(horiz, ((radius + 1, radius), (0, 0)), mode="reflect")
    cs = np.cumsum(padded, axis=0)
    result = (cs[ksize:, :] - cs[:-ksize, :]) / ksize
    return result


def _guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2013). Pure numpy, O(N).

    Parameters
    ----------
    guide : (H, W) float32 – edge guide image (e.g. grayscale reference).
    src   : (H, W) float32 – signal to filter (e.g. one warp channel).
    radius: int – window radius.
    eps   : float – regularization; smaller = more edge-preserving.

    Returns
    -------
    (H, W) float32 filtered output.
    """
    mean_g = _box_filter_2d(guide, radius)
    mean_s = _box_filter_2d(src, radius)
    corr_gg = _box_filter_2d(guide * guide, radius)
    corr_gs = _box_filter_2d(guide * src, radius)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = _box_filter_2d(a, radius)
    mean_b = _box_filter_2d(b, radius)

    return (mean_a * guide + mean_b).astype(np.float32)


def _rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 sRGB to CIE-LAB float32. No opencv needed."""
    rgb = rgb_uint8.astype(np.float32) / 255.0

    # Linearize sRGB
    mask = rgb > 0.04045
    linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Linear RGB -> XYZ (D65)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 white point
    x /= 0.95047
    z /= 1.08883

    # XYZ -> LAB
    def _f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.cbrt(t), t / (3.0 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a_ch = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return np.stack([L, a_ch, b_ch], axis=-1).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float32 CIE-LAB to uint8 sRGB. No opencv needed."""
    L, a_ch, b_ch = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a_ch / 500.0 + fy
    fz = fy - b_ch / 200.0

    delta = 6.0 / 29.0

    def _finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3.0 * delta**2 * (t - 4.0 / 29.0))

    x = 0.95047 * _finv(fx)
    y = _finv(fy)
    z = 1.08883 * _finv(fz)

    # XYZ -> linear RGB
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Gamma-compress to sRGB
    linear = np.stack([r, g, b], axis=-1).clip(0.0, 1.0)
    srgb = np.where(
        linear > 0.0031308,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return (srgb.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)


def _lum_rgb01(rgb_01: np.ndarray) -> np.ndarray:
    """Per-pixel luminance proxy used by Photoshop-style Color blend."""
    return (
        0.3 * rgb_01[..., 0] + 0.59 * rgb_01[..., 1] + 0.11 * rgb_01[..., 2]
    ).astype(np.float32)


def _clip_color_rgb01(rgb_01: np.ndarray) -> np.ndarray:
    """Gamut-safe clip preserving luminance (PDF/W3C blend-mode math)."""
    c = rgb_01.astype(np.float32, copy=True)
    l = _lum_rgb01(c)[..., None]

    n = c.min(axis=-1, keepdims=True)
    denom_low = np.maximum(l - n, 1e-6)
    c_low = l + ((c - l) * l) / denom_low
    c = np.where(n < 0.0, c_low, c)

    x = c.max(axis=-1, keepdims=True)
    denom_high = np.maximum(x - l, 1e-6)
    c_high = l + ((c - l) * (1.0 - l)) / denom_high
    c = np.where(x > 1.0, c_high, c)
    return np.clip(c, 0.0, 1.0)


def _set_lum_rgb01(rgb_01: np.ndarray, target_lum: np.ndarray) -> np.ndarray:
    current_lum = _lum_rgb01(rgb_01)
    shifted = rgb_01 + (target_lum - current_lum)[..., None]
    return _clip_color_rgb01(shifted)


def _photoshop_color_blend(
    *,
    base_rgb_uint8: np.ndarray,
    blend_rgb_uint8: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """Photoshop Color blend: keep base luminance, take blend hue/chroma."""
    base = base_rgb_uint8.astype(np.float32) / 255.0
    blend = blend_rgb_uint8.astype(np.float32) / 255.0
    opacity = float(np.clip(opacity, 0.0, 1.0))

    color_mode = _set_lum_rgb01(blend, _lum_rgb01(base))
    out = (1.0 - opacity) * base + opacity * color_mode
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _pil_to_tensor(img: Image.Image, *, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_chw_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0.0, 1.0).cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return (x * 255.0).round().astype(np.uint8)


def _save_chw_tensor_png(x: torch.Tensor, path: Path) -> None:
    Image.fromarray(_tensor_chw_to_uint8(x)).save(path)


def _overlap_to_color(overlap_01: np.ndarray) -> np.ndarray:
    overlap_01 = np.clip(overlap_01, 0.0, 1.0)
    r = overlap_01
    g = 1.0 - np.abs(2.0 * overlap_01 - 1.0)
    b = 1.0 - overlap_01
    rgb = np.stack([r, g, b], axis=-1)
    return (255.0 * rgb).round().astype(np.uint8)


def _draw_sampled_correspondences(
    *,
    ref_img: Image.Image,
    src_img: Image.Image,
    kpts_ref: np.ndarray,
    kpts_src: np.ndarray,
    certainties: np.ndarray,
    max_draw: int,
    save_path: Path,
    max_side: int = 0,
) -> int:
    ref_w, ref_h = ref_img.size
    src_w, src_h = src_img.size
    longest = max(ref_w + src_w, ref_h, src_h)
    scale = 1.0
    if max_side > 0 and longest > max_side:
        scale = max_side / float(longest)
        resample = _resample_bicubic()
        ref_img = ref_img.resize(
            (max(1, int(round(ref_w * scale))), max(1, int(round(ref_h * scale)))),
            resample=resample,
        )
        src_img = src_img.resize(
            (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
            resample=resample,
        )
        ref_w, ref_h = ref_img.size
        src_w, src_h = src_img.size

    canvas = Image.new("RGB", (ref_w + src_w, max(ref_h, src_h)), (255, 255, 255))
    canvas.paste(ref_img, (0, 0))
    canvas.paste(src_img, (ref_w, 0))
    draw = ImageDraw.Draw(canvas, mode="RGBA")

    n = int(kpts_ref.shape[0])
    if n == 0:
        canvas.save(save_path)
        return 0

    idx = np.arange(n)
    if n > max_draw:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_draw, replace=False)
    if certainties.size == n:
        idx = idx[np.argsort(certainties[idx])]

    for i in idx:
        x_ref, y_ref = float(kpts_ref[i, 0]) * scale, float(kpts_ref[i, 1]) * scale
        x_src, y_src = float(kpts_src[i, 0]) * scale, float(kpts_src[i, 1]) * scale
        x_src_shifted = x_src + ref_w

        c = float(certainties[i]) if certainties.size == n else 0.5
        c = max(0.0, min(1.0, c))
        color = (int((1.0 - c) * 255), int(c * 255), 64, 175)

        draw.line((x_ref, y_ref, x_src_shifted, y_src), fill=color, width=1)
        draw.ellipse((x_ref - 1, y_ref - 1, x_ref + 1, y_ref + 1), fill=(255, 64, 64, 220))
        draw.ellipse(
            (x_src_shifted - 1, y_src - 1, x_src_shifted + 1, y_src + 1),
            fill=(64, 128, 255, 220),
        )

    canvas.save(save_path)
    return int(idx.shape[0])


def _resample_bicubic() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _fit_within_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0:
        return w, h
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    scale = max_side / float(longest)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return out_w, out_h


def _resize_float2d_bilinear(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    t = torch.from_numpy(src.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    return out.numpy().astype(np.float32, copy=False)


def _guided_filter_with_mp_limit(
    *,
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
    max_megapixels: float,
) -> np.ndarray:
    h, w = guide.shape
    max_pixels = int(max(0.0, max_megapixels) * 1_000_000.0)
    if max_pixels <= 0 or (h * w) <= max_pixels:
        return _guided_filter(guide, src, radius, eps)

    scale = (max_pixels / float(h * w)) ** 0.5
    out_h = max(64, int(round(h * scale)))
    out_w = max(64, int(round(w * scale)))
    radius_ds = max(1, int(round(radius * scale)))

    guide_ds = _resize_float2d_bilinear(guide, out_h, out_w)
    src_ds = _resize_float2d_bilinear(src, out_h, out_w)
    filtered_ds = _guided_filter(guide_ds, src_ds, radius_ds, eps)
    return _resize_float2d_bilinear(filtered_ds, h, w)


def _mae_rgb_uint8(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0))


def _line_texture_1d(line: np.ndarray) -> float:
    if line.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(line.astype(np.float32)))))


def _scan_uniform_margin(
    *,
    border_mask: np.ndarray,
    luma: np.ndarray,
    side: str,
    max_crop: int,
    min_border_frac: float = 0.90,
    max_texture: float = 9.0,
) -> int:
    h, w = luma.shape
    margin = 0
    for i in range(max_crop):
        if side == "top":
            bm = border_mask[i, :]
            ll = luma[i, :]
        elif side == "bottom":
            bm = border_mask[h - 1 - i, :]
            ll = luma[h - 1 - i, :]
        elif side == "left":
            bm = border_mask[:, i]
            ll = luma[:, i]
        else:  # right
            bm = border_mask[:, w - 1 - i]
            ll = luma[:, w - 1 - i]

        if float(bm.mean()) < min_border_frac:
            break
        if _line_texture_1d(ll) > max_texture:
            break
        margin += 1
    return margin


def _auto_crop_border_like_margin(
    img: Image.Image,
    *,
    max_crop_frac: float = 0.22,
) -> tuple[Image.Image, dict[str, int], bool]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    if h < 128 or w < 128:
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False

    luma = rgb.mean(axis=2).astype(np.float32)
    sat = (rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32))

    rim_y = max(2, int(round(h * 0.02)))
    rim_x = max(2, int(round(w * 0.02)))
    rim_luma = np.concatenate(
        [luma[:rim_y, :].ravel(), luma[-rim_y:, :].ravel(), luma[:, :rim_x].ravel(), luma[:, -rim_x:].ravel()]
    )
    rim_sat = np.concatenate(
        [sat[:rim_y, :].ravel(), sat[-rim_y:, :].ravel(), sat[:, :rim_x].ravel(), sat[:, -rim_x:].ravel()]
    )
    rim_luma_median = float(np.median(rim_luma))
    rim_luma_mad = float(np.median(np.abs(rim_luma - rim_luma_median)))
    rim_sat_median = float(np.median(rim_sat))
    rim_luma_std = float(np.std(rim_luma))

    # Conservative guard: only attempt auto-crop when the edge appears mostly uniform.
    if rim_sat_median > 30.0 or rim_luma_std > 38.0:
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False

    luma_tol = max(16.0, 2.7 * rim_luma_mad)
    sat_limit = max(20.0, float(np.percentile(rim_sat, 85)) + 8.0)
    border_mask = (np.abs(luma - rim_luma_median) <= luma_tol) & (sat <= sat_limit)

    rim_mask = np.zeros((h, w), dtype=bool)
    rim_mask[:rim_y, :] = True
    rim_mask[-rim_y:, :] = True
    rim_mask[:, :rim_x] = True
    rim_mask[:, -rim_x:] = True
    rim_border_frac = float(border_mask[rim_mask].mean())
    if rim_border_frac < 0.70:
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False

    max_crop_x = max(0, int(round(w * max(0.0, min(0.45, max_crop_frac)))))
    max_crop_y = max(0, int(round(h * max(0.0, min(0.45, max_crop_frac)))))
    left = _scan_uniform_margin(border_mask=border_mask, luma=luma, side="left", max_crop=max_crop_x)
    right = _scan_uniform_margin(border_mask=border_mask, luma=luma, side="right", max_crop=max_crop_x)
    top = _scan_uniform_margin(border_mask=border_mask, luma=luma, side="top", max_crop=max_crop_y)
    bottom = _scan_uniform_margin(border_mask=border_mask, luma=luma, side="bottom", max_crop=max_crop_y)

    min_side_px = max(6, int(round(min(h, w) * 0.01)))
    sides = sum(int(v >= min_side_px) for v in (left, right, top, bottom))
    if sides < 2:
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False

    x0 = int(left)
    y0 = int(top)
    x1 = int(max(x0 + 1, w - right))
    y1 = int(max(y0 + 1, h - bottom))
    new_w = x1 - x0
    new_h = y1 - y0
    if new_w < int(0.5 * w) or new_h < int(0.5 * h):
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False
    if (left + right + top + bottom) < min_side_px:
        return img, {"left": 0, "right": 0, "top": 0, "bottom": 0}, False

    cropped = img.crop((x0, y0, x1, y1))
    return cropped, {"left": left, "right": right, "top": top, "bottom": bottom}, True


def _similarity_lstsq(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray | None:
    n = int(src_pts.shape[0])
    if n < 2:
        return None
    a = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)
    x = src_pts[:, 0].astype(np.float64)
    y = src_pts[:, 1].astype(np.float64)
    xp = dst_pts[:, 0].astype(np.float64)
    yp = dst_pts[:, 1].astype(np.float64)

    a[0::2, 0] = x
    a[0::2, 1] = -y
    a[0::2, 2] = 1.0
    a[1::2, 0] = y
    a[1::2, 1] = x
    a[1::2, 3] = 1.0
    b[0::2] = xp
    b[1::2] = yp

    try:
        params, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    aa, bb, tx, ty = [float(v) for v in params]
    m = np.array([[aa, -bb, tx], [bb, aa, ty]], dtype=np.float32)
    s = float(math.hypot(aa, bb))
    if not np.isfinite(s) or s < 1e-4 or s > 25.0:
        return None
    return m


def _apply_similarity_to_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    out_x = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    out_y = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return np.stack([out_x, out_y], axis=1)


def _estimate_similarity_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    iters: int,
    inlier_thresh_px: float,
) -> tuple[np.ndarray | None, float, float]:
    n = int(src_pts.shape[0])
    if n < 4:
        return None, 0.0, float("inf")

    rng = np.random.default_rng(0)
    best_m: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_median = float("inf")

    for _ in range(max(32, iters)):
        idx = rng.choice(n, size=2, replace=False)
        m = _similarity_lstsq(src_pts[idx], dst_pts[idx])
        if m is None:
            continue
        pred = _apply_similarity_to_points(m, src_pts)
        err = np.linalg.norm(pred - dst_pts, axis=1)
        inliers = err <= inlier_thresh_px
        count = int(inliers.sum())
        if count < 3:
            continue
        med = float(np.median(err[inliers]))
        if count > best_count or (count == best_count and med < best_median):
            best_count = count
            best_median = med
            best_m = m
            best_inliers = inliers

    if best_m is None or best_inliers is None:
        return None, 0.0, float("inf")

    refined = _similarity_lstsq(src_pts[best_inliers], dst_pts[best_inliers])
    if refined is None:
        refined = best_m

    pred = _apply_similarity_to_points(refined, src_pts)
    err = np.linalg.norm(pred - dst_pts, axis=1)
    inliers = err <= inlier_thresh_px
    inlier_ratio = float(inliers.mean())
    median_err = float(np.median(err[inliers])) if np.any(inliers) else float(np.median(err))
    return refined, inlier_ratio, median_err


def _warp_src_with_similarity_to_ref(
    src_img: Image.Image,
    *,
    ref_w: int,
    ref_h: int,
    src_to_ref: np.ndarray,
) -> Image.Image | None:
    src_rgb = np.asarray(src_img.convert("RGB"), dtype=np.float32) / 255.0
    src_h, src_w, _ = src_rgb.shape
    m3 = np.eye(3, dtype=np.float64)
    m3[:2, :] = src_to_ref.astype(np.float64)
    det = float(np.linalg.det(m3[:2, :2]))
    if abs(det) < 1e-8:
        return None
    ref_to_src = np.linalg.inv(m3)[:2, :]

    ys, xs = np.meshgrid(np.arange(ref_h, dtype=np.float32), np.arange(ref_w, dtype=np.float32), indexing="ij")
    x_src = ref_to_src[0, 0] * xs + ref_to_src[0, 1] * ys + ref_to_src[0, 2]
    y_src = ref_to_src[1, 0] * xs + ref_to_src[1, 1] * ys + ref_to_src[1, 2]

    x_norm = ((x_src + 0.5) / float(src_w)) * 2.0 - 1.0
    y_norm = ((y_src + 0.5) / float(src_h)) * 2.0 - 1.0
    grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)

    src_t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0)
    grid_t = torch.from_numpy(grid).unsqueeze(0)
    warped = F.grid_sample(
        src_t,
        grid_t,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]
    out_rgb = (warped.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_rgb)


def main() -> int:
    args = parse_args()
    run_start = perf_counter()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("[INFO] Runner bootstrap starting...", flush=True)
    print(f"[INFO] Python executable: {sys.executable}", flush=True)
    print("[INFO] Importing RoMa modules...", flush=True)
    try:
        from romav2 import RoMaV2
        from romav2.device import device as roma_device
    except Exception as exc:
        print(
            "[ERROR] Failed to import romav2. Install editable from this repo first, "
            "for example: uv pip install --python .\\.venv\\Scripts\\python.exe -e .",
            file=sys.stderr,
        )
        print(f"[ERROR] Import details: {exc}", file=sys.stderr)
        return 1
    print("[INFO] RoMa modules imported.", flush=True)

    if not args.ref.exists():
        print(f"[ERROR] --ref does not exist: {args.ref}", file=sys.stderr)
        return 1
    if not args.src.exists():
        print(f"[ERROR] --src does not exist: {args.src}", file=sys.stderr)
        return 1
    if args.num_samples <= 0:
        print("[ERROR] --num-samples must be > 0", file=sys.stderr)
        return 1
    if args.max_draw <= 0:
        print("[ERROR] --max-draw must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.regularize_overlap_thresh) < 1.0):
        print("[ERROR] --regularize-overlap-thresh must be in [0, 1).", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.color_opacity) <= 1.0):
        print("[ERROR] --color-opacity must be in [0, 1].", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.border_crop_max_frac) <= 0.45):
        print("[ERROR] --border-crop-max-frac must be in [0, 0.45].", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.min_global_inlier_ratio) <= 1.0):
        print("[ERROR] --min-global-inlier-ratio must be in [0, 1].", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.min_overlap_mean) <= 1.0):
        print("[ERROR] --min-overlap-mean must be in [0, 1].", file=sys.stderr)
        return 1
    if not (0.0 <= float(args.auto_reference_fallback_overlap) <= 1.0):
        print("[ERROR] --auto-reference-fallback-overlap must be in [0, 1].", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {args.outdir.resolve()}")

    ref_img = Image.open(args.ref).convert("RGB")
    src_img = Image.open(args.src).convert("RGB")
    ref_orig_w, ref_orig_h = ref_img.size
    src_orig_w, src_orig_h = src_img.size
    ref_match_path = args.ref
    src_match_path = args.src

    preprocessing: dict[str, object] = {
        "auto_border_crop_enabled": not bool(args.disable_auto_border_crop),
        "ref_crop_applied": False,
        "src_crop_applied": False,
        "ref_crop_margins_lrtb": [0, 0, 0, 0],
        "src_crop_margins_lrtb": [0, 0, 0, 0],
        "global_prealign_enabled": not bool(args.disable_global_prealign),
        "global_prealign_applied": False,
        "global_prealign_scale": None,
        "global_prealign_rotation_deg": None,
        "global_prealign_inlier_ratio": None,
        "global_prealign_median_err_px": None,
        "global_prealign_artifact": None,
    }

    if not args.disable_auto_border_crop:
        max_crop_frac = float(max(0.0, min(0.45, args.border_crop_max_frac)))
        ref_cropped, ref_margins, ref_applied = _auto_crop_border_like_margin(
            ref_img,
            max_crop_frac=max_crop_frac,
        )
        if ref_applied:
            ref_img = ref_cropped
            ref_match_path = args.outdir / "ref_autocropped.png"
            ref_img.save(ref_match_path)
            preprocessing["ref_crop_applied"] = True
            preprocessing["ref_crop_margins_lrtb"] = [
                int(ref_margins["left"]),
                int(ref_margins["right"]),
                int(ref_margins["top"]),
                int(ref_margins["bottom"]),
            ]
            print(
                "[INFO] Auto-cropped border from ref "
                f"(L{ref_margins['left']} R{ref_margins['right']} T{ref_margins['top']} B{ref_margins['bottom']})."
            )

        src_cropped, src_margins, src_applied = _auto_crop_border_like_margin(
            src_img,
            max_crop_frac=max_crop_frac,
        )
        if src_applied:
            src_img = src_cropped
            src_match_path = args.outdir / "src_autocropped.png"
            src_img.save(src_match_path)
            preprocessing["src_crop_applied"] = True
            preprocessing["src_crop_margins_lrtb"] = [
                int(src_margins["left"]),
                int(src_margins["right"]),
                int(src_margins["top"]),
                int(src_margins["bottom"]),
            ]
            print(
                "[INFO] Auto-cropped border from src "
                f"(L{src_margins['left']} R{src_margins['right']} T{src_margins['top']} B{src_margins['bottom']})."
            )

    ref_w, ref_h = ref_img.size
    src_w, src_h = src_img.size
    print(f"[INFO] Loaded images. ref={ref_w}x{ref_h}, src={src_w}x{src_h}")
    ref_mp = (ref_w * ref_h) / 1_000_000.0
    src_mp = (src_w * src_h) / 1_000_000.0
    print(
        f"[INFO] Input megapixels: ref={ref_mp:.2f}MP, src={src_mp:.2f}MP "
        f"(diag-max-side={args.diag_max_side}, filter-max-mp={args.filter_max_megapixels:.2f})"
    )
    if max(ref_mp, src_mp) >= 20.0:
        print(
            "[WARN] Large-image run detected. Full-resolution diagnostics can be extremely slow; "
            "this run uses downscaled diagnostics and guided-filter MP caps."
        )

    torch.set_float32_matmul_precision("highest")
    cuda_available = torch.cuda.is_available()
    torch_build = str(torch.__version__)
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "Unknown CUDA device"
        print(f"[INFO] Runtime device: CUDA ({device_name})")
    else:
        build_hint = "CPU-only torch build detected" if "+cpu" in torch_build else "CUDA runtime unavailable"
        print(
            "[WARN] Runtime device: CPU only "
            f"({build_hint}; torch={torch_build}). Dense match is the main bottleneck and may take many minutes."
        )
        if max(ref_mp, src_mp) >= 3.0:
            print(
                "[WARN] For faster runs use a CUDA-enabled PyTorch build + NVIDIA GPU, "
                "or lower quality preset/image size."
            )

    stage_start = perf_counter()
    print(f"[INFO] Initializing RoMa v2 (setting={args.setting}, compile={args.compile}) ...")
    model = RoMaV2(RoMaV2.Cfg(setting=args.setting, compile=args.compile))
    print(f"[TIMING] Model init: {perf_counter() - stage_start:.2f}s")

    # Capture model config before it's potentially freed for VRAM
    model_info = {
        "setting": args.setting,
        "compile": bool(args.compile),
        "H_lr": int(model.H_lr),
        "W_lr": int(model.W_lr),
        "H_hr": int(model.H_hr) if model.H_hr is not None else None,
        "W_hr": int(model.W_hr) if model.W_hr is not None else None,
        "bidirectional": bool(model.bidirectional),
        "threshold": float(model.threshold) if model.threshold is not None else None,
        "balanced_sampling": bool(model.balanced_sampling),
    }

    if args.warmup_only:
        stage_start = perf_counter()
        print("[INFO] Warmup-only mode: running one dense match and exiting early.")
        _ = model.match(str(ref_match_path), str(src_match_path))
        print(f"[TIMING] Warmup match: {perf_counter() - stage_start:.2f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[TIMING] Warmup-only total: {perf_counter() - run_start:.2f}s")
        return 0

    stage_start = perf_counter()
    print("[INFO] Running dense match (ref -> src) ...")
    preds = model.match(str(ref_match_path), str(src_match_path))
    print("[INFO] Match complete.")
    print(f"[TIMING] Dense match: {perf_counter() - stage_start:.2f}s")

    output_files: list[Path] = []
    overlap_mean: float | None = None
    overlap_median: float | None = None
    effective_regularize_fallback = str(args.regularize_fallback)
    auto_reference_fallback_applied = False
    if not args.disable_global_prealign:
        prealign_stage_start = perf_counter()
        print("[INFO] Estimating global pre-alignment (scale/rotation/translation) ...")
        try:
            prealign_samples = min(max(1200, int(args.num_samples)), 5000)
            matches0, sampled_overlaps0, _pa, _pb = model.sample(preds, prealign_samples)
            kpts_ref0, kpts_src0 = model.to_pixel_coordinates(matches0, ref_h, ref_w, src_h, src_w)
            src_pts = kpts_src0.detach().cpu().numpy()
            ref_pts = kpts_ref0.detach().cpu().numpy()
            conf = sampled_overlaps0.detach().cpu().numpy().reshape(-1)

            if conf.size > 0:
                q = float(np.quantile(conf, 0.60))
                keep = conf >= q
                if int(np.sum(keep)) >= 60:
                    src_pts = src_pts[keep]
                    ref_pts = ref_pts[keep]

            diag = float(math.hypot(ref_w, ref_h))
            inlier_thresh = max(4.0, 0.006 * diag)
            src_to_ref_m, inlier_ratio, median_err = _estimate_similarity_ransac(
                src_pts.astype(np.float32, copy=False),
                ref_pts.astype(np.float32, copy=False),
                iters=450,
                inlier_thresh_px=inlier_thresh,
            )
            preprocessing["global_prealign_inlier_ratio"] = float(inlier_ratio)
            preprocessing["global_prealign_median_err_px"] = float(median_err)

            if src_to_ref_m is not None:
                a = float(src_to_ref_m[0, 0])
                b = float(src_to_ref_m[1, 0])
                scale = float(math.hypot(a, b))
                rot_deg = float(math.degrees(math.atan2(b, a)))
                preprocessing["global_prealign_scale"] = scale
                preprocessing["global_prealign_rotation_deg"] = rot_deg
                print(
                    "[INFO] Global pre-align estimate: "
                    f"scale={scale:.4f}, rot={rot_deg:.2f} deg, "
                    f"inliers={inlier_ratio:.3f}, median_err={median_err:.2f}px."
                )

                if inlier_ratio >= float(args.min_global_inlier_ratio) and 0.30 <= scale <= 3.50:
                    prealigned = _warp_src_with_similarity_to_ref(
                        src_img,
                        ref_w=ref_w,
                        ref_h=ref_h,
                        src_to_ref=src_to_ref_m,
                    )
                    if prealigned is not None:
                        src_img = prealigned
                        src_w, src_h = src_img.size
                        src_match_path = args.outdir / "src_global_prealigned.png"
                        src_img.save(src_match_path)
                        output_files.append(src_match_path)
                        preprocessing["global_prealign_applied"] = True
                        preprocessing["global_prealign_artifact"] = str(src_match_path.resolve())
                        print("[INFO] Running dense match on globally pre-aligned source ...")
                        stage_match2 = perf_counter()
                        preds = model.match(str(ref_match_path), str(src_match_path))
                        print("[INFO] Second match complete (after global pre-align).")
                        print(f"[TIMING] Dense match (pre-aligned): {perf_counter() - stage_match2:.2f}s")
                    else:
                        print("[WARN] Global pre-align transform could not be applied. Continuing without it.")
                else:
                    print(
                        "[WARN] Global pre-align rejected due to low inlier quality "
                        f"(inliers={inlier_ratio:.3f}, required>={float(args.min_global_inlier_ratio):.3f})."
                    )
            else:
                print("[WARN] Could not estimate a stable global similarity transform.")
        except Exception as exc:
            print(f"[WARN] Global pre-align failed: {exc}")
        print(f"[TIMING] Global pre-alignment stage: {perf_counter() - prealign_stage_start:.2f}s")

    pred_shapes: dict[str, dict[str, object] | None] = {}
    stage_start = perf_counter()
    if args.save_dense_preds:
        print("[INFO] Saving dense outputs (.npy) ...")
    else:
        print("[INFO] Skipping dense .npy dumps (enable with --save-dense-preds).")
    for key, value in preds.items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if args.save_dense_preds:
                npy_path = args.outdir / f"{key}.npy"
                np.save(npy_path, array)
                output_files.append(npy_path)
            pred_shapes[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        else:
            pred_shapes[key] = None
    print(f"[TIMING] Pred export/metadata: {perf_counter() - stage_start:.2f}s")

    warp_ab = preds.get("warp_AB")
    if not isinstance(warp_ab, torch.Tensor):
        print("[ERROR] preds['warp_AB'] missing or invalid.", file=sys.stderr)
        return 1
    warp_ab = warp_ab[0]

    overlap_ab = preds.get("overlap_AB")
    overlap_small: torch.Tensor | None = None
    overlap_full: torch.Tensor | None = None
    stage_start = perf_counter()
    if isinstance(overlap_ab, torch.Tensor):
        overlap_small = overlap_ab[0, ..., 0]
        overlap_small_np = overlap_small.detach().cpu().numpy()
        overlap_gray = (255.0 * np.clip(overlap_small_np, 0.0, 1.0)).round().astype(np.uint8)
        overlap_gray_path = args.outdir / "overlap_AB_gray.png"
        Image.fromarray(overlap_gray).save(overlap_gray_path)
        output_files.append(overlap_gray_path)

        overlap_full = F.interpolate(
            overlap_small.unsqueeze(0).unsqueeze(0),
            size=(ref_h, ref_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        vis_w, vis_h = _fit_within_size(ref_w, ref_h, args.diag_max_side)
        if (vis_w, vis_h) != (ref_w, ref_h):
            overlap_vis = F.interpolate(
                overlap_full.unsqueeze(0).unsqueeze(0),
                size=(vis_h, vis_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            overlap_vis_np = overlap_vis.detach().cpu().numpy()
            ref_vis = ref_img.resize((vis_w, vis_h), resample=_resample_bicubic())
            ref_np = np.asarray(ref_vis, dtype=np.float32)
        else:
            overlap_vis_np = overlap_full.detach().cpu().numpy()
            ref_np = np.asarray(ref_img, dtype=np.float32)
        overlap_color_np = _overlap_to_color(overlap_vis_np)

        overlap_color_path = args.outdir / "overlap_AB_color_refsize.png"
        Image.fromarray(overlap_color_np).save(overlap_color_path)
        output_files.append(overlap_color_path)

        overlap_overlay = (
            0.65 * ref_np + 0.35 * overlap_color_np.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        overlap_overlay_path = args.outdir / "overlap_AB_overlay_on_ref.png"
        Image.fromarray(overlap_overlay).save(overlap_overlay_path)
        output_files.append(overlap_overlay_path)
    print(f"[TIMING] Overlap visualization: {perf_counter() - stage_start:.2f}s")

    if overlap_full is not None:
        overlap_mean = float(overlap_full.mean().item())
        overlap_median = float(overlap_full.median().item())
        print(
            "[INFO] Final overlap stats: "
            f"mean={overlap_mean:.4f}, median={overlap_median:.4f}."
        )

        if (
            args.regularize_fallback == "identity"
            and overlap_mean < float(args.auto_reference_fallback_overlap)
        ):
            effective_regularize_fallback = "reference"
            auto_reference_fallback_applied = True
            print(
                "[WARN] Low overlap detected. Auto-switching regularize fallback "
                f"identity -> reference (mean={overlap_mean:.4f}, "
                f"threshold={float(args.auto_reference_fallback_overlap):.4f})."
            )
        else:
            print(
                "[INFO] Regularize fallback in use: "
                f"{effective_regularize_fallback}."
            )

        if overlap_mean < float(args.min_overlap_mean) and not bool(args.allow_low_overlap):
            guardrail_path = args.outdir / "guardrail_failure.txt"
            guardrail_path.write_text(
                (
                    "Low-overlap guardrail triggered.\n"
                    f"overlap_mean: {overlap_mean:.6f}\n"
                    f"required_min_overlap_mean: {float(args.min_overlap_mean):.6f}\n"
                    f"allow_low_overlap: {bool(args.allow_low_overlap)}\n"
                    f"requested_regularize_fallback: {args.regularize_fallback}\n"
                    f"effective_regularize_fallback: {effective_regularize_fallback}\n"
                ),
                encoding="utf-8",
            )
            output_files.append(guardrail_path)
            print(
                "[ERROR] Guardrail: overlap mean below threshold "
                f"({overlap_mean:.4f} < {float(args.min_overlap_mean):.4f}). "
                "Aborting run. Use --allow-low-overlap to override."
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 2
    else:
        print("[WARN] overlap_AB is unavailable; low-overlap guardrail skipped.")
        print(f"[INFO] Regularize fallback in use: {effective_regularize_fallback}.")

    stage_start = perf_counter()
    print("[INFO] Building source->reference warp image ...")
    warp_ab_full = F.interpolate(
        warp_ab.permute(2, 0, 1).unsqueeze(0),
        size=(ref_h, ref_w),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    src_tensor = _pil_to_tensor(src_img, device=roma_device)
    ref_tensor = _pil_to_tensor(ref_img, device=roma_device)[0]

    # --- Build confidence mask from overlap scores ---
    confidence_mask: torch.Tensor | None = None
    if overlap_full is not None:
        conf_thresh = float(args.regularize_overlap_thresh)
        ramp_width = 0.15
        confidence_mask = torch.clamp(
            (overlap_full - conf_thresh) / ramp_width, 0.0, 1.0
        )
        conf_pct = float(confidence_mask.mean().item()) * 100
        print(f"[INFO] Confidence mask: {conf_pct:.1f}% of pixels above threshold ({conf_thresh:.2f})")

        conf_vis = confidence_mask
        vis_w, vis_h = _fit_within_size(ref_w, ref_h, args.diag_max_side)
        if (vis_w, vis_h) != (ref_w, ref_h):
            conf_vis = F.interpolate(
                confidence_mask.unsqueeze(0).unsqueeze(0),
                size=(vis_h, vis_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        conf_mask_np = (conf_vis.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        conf_mask_path = args.outdir / "confidence_mask.png"
        Image.fromarray(conf_mask_np).save(conf_mask_path)
        output_files.append(conf_mask_path)
    else:
        print("[WARN] No overlap scores available, skipping confidence masking.")

    # Save raw warp output (before any smoothing)
    warped_src = F.grid_sample(
        src_tensor,
        warp_ab_full.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]
    warped_src_path = args.outdir / "warped_src_to_ref.png"
    _save_chw_tensor_png(warped_src, warped_src_path)
    output_files.append(warped_src_path)
    print(f"[TIMING] Raw warp build/save: {perf_counter() - stage_start:.2f}s")

    # --- Guided-filter warp smoothing ---
    # Pipeline order: smooth RAW warp → THEN apply confidence mask (once).
    gf_radius = int(args.guided_filter_radius)
    warped_src_smooth: torch.Tensor | None = None
    if gf_radius > 0:
        stage_start = perf_counter()
        print(
            f"[INFO] Applying guided-filter smoothing to warp field "
            f"(radius={gf_radius}, eps={args.guided_filter_eps:.1e}) ..."
        )
        ref_gray = np.asarray(ref_img.convert("L"), dtype=np.float32) / 255.0
        warp_np = warp_ab_full.detach().cpu().numpy()  # (H, W, 2)

        # Smooth the RAW warp (not the confidence-masked one)
        warp_smooth_np = np.stack(
            [
                _guided_filter_with_mp_limit(
                    guide=ref_gray,
                    src=warp_np[..., 0],
                    radius=gf_radius,
                    eps=args.guided_filter_eps,
                    max_megapixels=float(args.filter_max_megapixels),
                ),
                _guided_filter_with_mp_limit(
                    guide=ref_gray,
                    src=warp_np[..., 1],
                    radius=gf_radius,
                    eps=args.guided_filter_eps,
                    max_megapixels=float(args.filter_max_megapixels),
                ),
            ],
            axis=-1,
        )
        warp_smooth_full = torch.from_numpy(warp_smooth_np).to(roma_device)

        # Apply confidence mask ONCE: blend smoothed warp with the raw RoMa warp
        # in low-confidence regions. Blending with identity caused visible
        # double-exposure artifacts when src/ref geometry differs.
        if confidence_mask is not None:
            mask_2d = confidence_mask.unsqueeze(-1)
            warp_smooth_full = mask_2d * warp_smooth_full + (1.0 - mask_2d) * warp_ab_full

        # Keep sampling coordinates in the valid grid_sample range.
        warp_smooth_full = warp_smooth_full.clamp(-1.0, 1.0)

        warped_src_smooth = F.grid_sample(
            src_tensor,
            warp_smooth_full.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
        warped_smooth_path = args.outdir / "warped_src_smooth.png"
        _save_chw_tensor_png(warped_src_smooth, warped_smooth_path)
        output_files.append(warped_smooth_path)
        print("[INFO] Saved guided-filter smoothed warp.")
        print(f"[TIMING] Warp guided-filter stage: {perf_counter() - stage_start:.2f}s")
    else:
        print("[INFO] Guided-filter smoothing disabled (radius=0).")

    # --- Sampling correspondences (must happen before model is freed) ---
    stage_start = perf_counter()
    print("[INFO] Sampling correspondences for visualization ...")
    h_map = int(warp_ab.shape[0])
    w_map = int(warp_ab.shape[1])
    bidirectional = isinstance(preds.get("warp_BA"), torch.Tensor)
    sample_pool = h_map * w_map * (2 if bidirectional else 1)
    max_safe_samples = max(1, sample_pool // 4)
    num_samples = min(int(args.num_samples), max_safe_samples)

    matches, sampled_overlaps, precision_ab_s, precision_ba_s = model.sample(preds, num_samples)
    matches_path = args.outdir / "sampled_matches_normalized.npy"
    np.save(matches_path, matches.detach().cpu().numpy())
    output_files.append(matches_path)

    sampled_overlap_path = args.outdir / "sampled_overlaps.npy"
    np.save(sampled_overlap_path, sampled_overlaps.detach().cpu().numpy())
    output_files.append(sampled_overlap_path)

    if precision_ab_s is not None:
        precision_ab_path = args.outdir / "sampled_precision_AB.npy"
        np.save(precision_ab_path, precision_ab_s.detach().cpu().numpy())
        output_files.append(precision_ab_path)
    if precision_ba_s is not None:
        precision_ba_path = args.outdir / "sampled_precision_BA.npy"
        np.save(precision_ba_path, precision_ba_s.detach().cpu().numpy())
        output_files.append(precision_ba_path)

    kpts_ref, kpts_src = model.to_pixel_coordinates(matches, ref_h, ref_w, src_h, src_w)
    kpts_ref_np = kpts_ref.detach().cpu().numpy()
    kpts_src_np = kpts_src.detach().cpu().numpy()
    sampled_overlaps_np = sampled_overlaps.detach().cpu().numpy().reshape(-1)

    corr_vis_path = args.outdir / "sampled_correspondences.png"
    drawn = _draw_sampled_correspondences(
        ref_img=ref_img,
        src_img=src_img,
        kpts_ref=kpts_ref_np,
        kpts_src=kpts_src_np,
        certainties=sampled_overlaps_np,
        max_draw=args.max_draw,
        save_path=corr_vis_path,
        max_side=int(args.diag_max_side),
    )
    output_files.append(corr_vis_path)
    print(f"[TIMING] Sampling/correspondence vis: {perf_counter() - stage_start:.2f}s")

    # Free RoMa model to reclaim VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Photoshop-style Color blend transfer ---
    stage_start = perf_counter()
    final_rgb: np.ndarray | None = None
    if not args.no_color_transfer:
        # Use the best available warped image: smooth > raw
        color_donor = warped_src_smooth if warped_src_smooth is not None else warped_src
        donor_rgb = _tensor_chw_to_uint8(color_donor)
        ref_rgb = np.asarray(ref_img.convert("RGB"), dtype=np.uint8)

        chroma_radius = int(args.chroma_filter_radius)
        if chroma_radius > 0:
            # Smooth donor chroma with reference luminance as guide.
            ref_gray = np.asarray(ref_img.convert("L"), dtype=np.float32) / 255.0
            donor_lab = _rgb_to_lab(donor_rgb)
            donor_a = _guided_filter_with_mp_limit(
                guide=ref_gray,
                src=donor_lab[..., 1].astype(np.float32),
                radius=chroma_radius,
                eps=args.guided_filter_eps,
                max_megapixels=float(args.filter_max_megapixels),
            )
            donor_b = _guided_filter_with_mp_limit(
                guide=ref_gray,
                src=donor_lab[..., 2].astype(np.float32),
                radius=chroma_radius,
                eps=args.guided_filter_eps,
                max_megapixels=float(args.filter_max_megapixels),
            )
            donor_lab[..., 1] = donor_a
            donor_lab[..., 2] = donor_b
            donor_rgb = _lab_to_rgb(donor_lab)
            print(f"[INFO] Chrominance guided filter: radius={chroma_radius}")

        final_rgb = _photoshop_color_blend(
            base_rgb_uint8=ref_rgb,
            blend_rgb_uint8=donor_rgb,
            opacity=float(args.color_opacity),
        )

        final_colorized_path = args.outdir / "final_colorized.png"
        Image.fromarray(final_rgb).save(final_colorized_path)
        output_files.append(final_colorized_path)

        print(
            "[INFO] Saved Photoshop-style Color blend result "
            f"(opacity={float(args.color_opacity):.2f}) -> final_colorized.png"
        )
    else:
        print("[INFO] Color transfer disabled (--no-color-transfer).")
    print(f"[TIMING] Color transfer stage: {perf_counter() - stage_start:.2f}s")
    in_bounds = warp_ab_full.abs().amax(dim=-1, keepdim=True).le(1.0).float()
    if overlap_full is not None:
        validity = overlap_full.unsqueeze(-1) * in_bounds
    else:
        validity = in_bounds
    validity_chw = validity.permute(2, 0, 1)
    warped_src_masked = warped_src * validity_chw + (1.0 - validity_chw)
    warped_src_masked_path = args.outdir / "warped_src_to_ref_masked_by_overlap.png"
    _save_chw_tensor_png(warped_src_masked, warped_src_masked_path)
    output_files.append(warped_src_masked_path)

    regularized_output_path: Path | None = None
    warped_regularized_rgb: np.ndarray | None = None
    if overlap_full is not None and effective_regularize_fallback != "none":
        tau = float(args.regularize_overlap_thresh)
        denom = max(1e-6, 1.0 - tau)
        alpha = ((overlap_full - tau) / denom).clamp(0.0, 1.0).unsqueeze(0)

        if effective_regularize_fallback == "identity":
            identity_theta = torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                dtype=src_tensor.dtype,
                device=src_tensor.device,
            )
            identity_grid = F.affine_grid(
                identity_theta,
                size=(1, 3, ref_h, ref_w),
                align_corners=False,
            )
            fallback_img = F.grid_sample(
                src_tensor,
                identity_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0]
        elif effective_regularize_fallback == "reference":
            fallback_img = ref_tensor
        else:
            fallback_img = warped_src

        warped_regularized = alpha * warped_src + (1.0 - alpha) * fallback_img
        warped_regularized_rgb = _tensor_chw_to_uint8(warped_regularized)
        regularized_output_path = args.outdir / "warped_src_to_ref_regularized.png"
        _save_chw_tensor_png(warped_regularized, regularized_output_path)
        output_files.append(regularized_output_path)

        alpha_img = (alpha.squeeze(0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        alpha_path = args.outdir / "regularization_alpha.png"
        Image.fromarray(alpha_img).save(alpha_path)
        output_files.append(alpha_path)

    stage_start = perf_counter()
    print("[INFO] Building before/after diagnostic images ...")
    resample = _resample_bicubic()
    ref_rgb_full = np.asarray(ref_img.convert("RGB"), dtype=np.uint8)
    src_resized_to_ref = src_img.resize((ref_w, ref_h), resample=resample)
    src_resized_rgb_full = np.asarray(src_resized_to_ref.convert("RGB"), dtype=np.uint8)
    warped_src_rgb = _tensor_chw_to_uint8(warped_src)
    warped_src_smooth_rgb = _tensor_chw_to_uint8(warped_src_smooth) if warped_src_smooth is not None else None

    diag_w, diag_h = _fit_within_size(ref_w, ref_h, int(args.diag_max_side))

    def _diag_resize(rgb: np.ndarray) -> np.ndarray:
        if (diag_w, diag_h) == (ref_w, ref_h):
            return rgb
        return np.asarray(
            Image.fromarray(rgb).resize((diag_w, diag_h), resample=resample),
            dtype=np.uint8,
        )

    ref_diag = _diag_resize(ref_rgb_full)
    src_diag = _diag_resize(src_resized_rgb_full)
    warped_diag = _diag_resize(warped_src_rgb)
    warped_smooth_diag = _diag_resize(warped_src_smooth_rgb) if warped_src_smooth_rgb is not None else None
    warped_regularized_diag = _diag_resize(warped_regularized_rgb) if warped_regularized_rgb is not None else None
    final_diag = _diag_resize(final_rgb) if final_rgb is not None else None

    before_after_triptych = np.concatenate((ref_diag, src_diag, warped_diag), axis=1)
    before_after_triptych_path = args.outdir / "before_after_triptych.png"
    Image.fromarray(before_after_triptych).save(before_after_triptych_path)
    output_files.append(before_after_triptych_path)

    def _diff_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        g = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2, keepdims=True)
        g = np.clip(g, 0.0, 255.0).astype(np.uint8)
        return np.repeat(g, 3, axis=2)

    diff_panels = [_diff_rgb(ref_diag, src_diag), _diff_rgb(ref_diag, warped_diag)]
    if warped_regularized_diag is not None:
        diff_panels.append(_diff_rgb(ref_diag, warped_regularized_diag))
    if final_diag is not None:
        diff_panels.append(_diff_rgb(ref_diag, final_diag))
    diff_panel = np.concatenate(diff_panels, axis=1)
    diff_panel_path = args.outdir / "difference_before_after.png"
    Image.fromarray(diff_panel).save(diff_panel_path)
    output_files.append(diff_panel_path)

    mae_before = _mae_rgb_uint8(ref_diag, src_diag)
    mae_warped = _mae_rgb_uint8(ref_diag, warped_diag)
    mae_warped_regularized = (
        _mae_rgb_uint8(ref_diag, warped_regularized_diag)
        if warped_regularized_diag is not None
        else None
    )
    mae_smooth_warped = (
        _mae_rgb_uint8(ref_diag, warped_smooth_diag)
        if warped_smooth_diag is not None
        else None
    )
    mae_final_colorized = (
        _mae_rgb_uint8(ref_diag, final_diag)
        if final_diag is not None
        else None
    )
    print(
        f"[TIMING] Diagnostic image build/metrics: {perf_counter() - stage_start:.2f}s "
        f"(diag size {diag_w}x{diag_h})"
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "roma_device": str(roma_device),
        "inputs": {
            "ref_path": str(args.ref.resolve()),
            "src_path": str(args.src.resolve()),
            "ref_original_size_wh": [int(ref_orig_w), int(ref_orig_h)],
            "src_original_size_wh": [int(src_orig_w), int(src_orig_h)],
            "ref_match_size_wh": [int(ref_w), int(ref_h)],
            "src_match_size_wh": [int(src_w), int(src_h)],
        },
        "model": model_info,
        "preprocessing": preprocessing,
        "regularization": {
            "fallback_requested": args.regularize_fallback,
            "fallback_effective": effective_regularize_fallback,
            "fallback_auto_switched_to_reference": bool(auto_reference_fallback_applied),
            "overlap_threshold": float(args.regularize_overlap_thresh),
            "auto_reference_fallback_overlap": float(args.auto_reference_fallback_overlap),
            "regularized_output_written": regularized_output_path is not None,
        },
        "overlap_guardrail": {
            "final_overlap_mean": overlap_mean,
            "final_overlap_median": overlap_median,
            "min_overlap_mean": float(args.min_overlap_mean),
            "allow_low_overlap": bool(args.allow_low_overlap),
        },
        "guided_filter": {
            "radius": gf_radius,
            "eps": float(args.guided_filter_eps),
            "enabled": gf_radius > 0,
        },
        "color_transfer": {
            "enabled": not args.no_color_transfer,
            "method": "Photoshop-style Color blend (base luminance + donor color)",
            "chroma_filter_radius": int(args.chroma_filter_radius),
            "opacity": float(args.color_opacity),
        },
        "performance": {
            "diag_max_side": int(args.diag_max_side),
            "filter_max_megapixels": float(args.filter_max_megapixels),
            "save_dense_preds": bool(args.save_dense_preds),
        },
        "metrics": {
            "computed_on": [int(diag_w), int(diag_h)],
            "mae_before_src_resized_vs_ref": mae_before,
            "mae_after_warped_vs_ref": mae_warped,
            "mae_after_regularized_vs_ref": mae_warped_regularized,
            "mae_after_smooth_warped_vs_ref": mae_smooth_warped,
            "mae_after_final_colorized_vs_ref": mae_final_colorized,
        },
        "preds": pred_shapes,
        "sampling": {
            "requested_num_samples": int(args.num_samples),
            "num_samples_used": int(num_samples),
            "max_draw": int(args.max_draw),
            "num_drawn": int(drawn),
            "sample_pool_size": int(sample_pool),
            "warp_map_hw": [h_map, w_map],
        },
        "warp_strategy": {
            "implemented": True,
            "details": [
                "Used preds['warp_AB'] from model.match(ref, src).",
                "warp_AB is treated as normalized sampling coordinates in source image space (same approach as demo/demo_match.py).",
                "warp_AB was bilinearly upsampled from map resolution to reference image resolution before torch.nn.functional.grid_sample.",
                "warped_src_to_ref_masked_by_overlap.png blends overlap/in-bounds validity for visual quality assessment.",
                "warped_src_to_ref_regularized.png (if enabled) blends dense warp with fallback content in low-overlap regions to reduce unstable distortions.",
                "Coordinates are normalized and learned on RoMa's internal resized inputs; absolute pixel interpretation on original images is approximate if aspect ratios differ strongly.",
            ],
        },
        "artifacts": [str(p.resolve()) for p in output_files],
    }

    summary_json_path = args.outdir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_files.append(summary_json_path)

    summary_txt_lines = [
        f"timestamp_utc: {summary['timestamp_utc']}",
        f"python_executable: {summary['python_executable']}",
        f"torch_version: {summary['torch_version']}",
        f"cuda_available: {summary['cuda_available']}",
        f"cuda_device_name: {summary['cuda_device_name']}",
        f"roma_device: {summary['roma_device']}",
        (
            f"ref: {summary['inputs']['ref_path']} "
            f"(orig {ref_orig_w}x{ref_orig_h}, match {ref_w}x{ref_h})"
        ),
        (
            f"src: {summary['inputs']['src_path']} "
            f"(orig {src_orig_w}x{src_orig_h}, match {src_w}x{src_h})"
        ),
        f"setting: {summary['model']['setting']}",
        f"compile: {summary['model']['compile']}",
        f"warp_map_hw: {h_map}x{w_map}",
        f"num_samples_requested: {args.num_samples}",
        f"num_samples_used: {num_samples}",
        f"num_drawn_correspondences: {drawn}",
        f"regularize_fallback_requested: {args.regularize_fallback}",
        f"regularize_fallback_effective: {effective_regularize_fallback}",
        f"regularize_fallback_auto_switched_to_reference: {bool(auto_reference_fallback_applied)}",
        f"regularize_overlap_thresh: {args.regularize_overlap_thresh}",
        (
            f"final_overlap_mean: {overlap_mean:.6f}"
            if overlap_mean is not None
            else "final_overlap_mean: n/a"
        ),
        (
            f"final_overlap_median: {overlap_median:.6f}"
            if overlap_median is not None
            else "final_overlap_median: n/a"
        ),
        f"min_overlap_mean: {float(args.min_overlap_mean):.6f}",
        f"allow_low_overlap: {bool(args.allow_low_overlap)}",
        f"auto_reference_fallback_overlap: {float(args.auto_reference_fallback_overlap):.6f}",
        f"guided_filter_radius: {gf_radius}",
        f"guided_filter_eps: {args.guided_filter_eps}",
        f"color_transfer: {'enabled' if not args.no_color_transfer else 'disabled'}",
        f"color_opacity: {float(args.color_opacity):.3f}",
        f"diag_max_side: {int(args.diag_max_side)}",
        f"filter_max_megapixels: {float(args.filter_max_megapixels):.2f}",
        f"save_dense_preds: {bool(args.save_dense_preds)}",
        f"metrics_computed_on: {diag_w}x{diag_h}",
        f"mae_before_src_resized_vs_ref: {mae_before:.6f}",
        f"mae_after_warped_vs_ref: {mae_warped:.6f}",
        (
            f"mae_after_regularized_vs_ref: {mae_warped_regularized:.6f}"
            if mae_warped_regularized is not None
            else "mae_after_regularized_vs_ref: n/a"
        ),
        (
            f"mae_after_smooth_warped_vs_ref: {mae_smooth_warped:.6f}"
            if mae_smooth_warped is not None
            else "mae_after_smooth_warped_vs_ref: n/a"
        ),
        (
            f"mae_after_final_colorized_vs_ref: {mae_final_colorized:.6f}"
            if mae_final_colorized is not None
            else "mae_after_final_colorized_vs_ref: n/a"
        ),
        "",
        (
            "preprocessing_auto_border_crop: "
            f"{bool(preprocessing['auto_border_crop_enabled'])}"
        ),
        (
            "preprocessing_ref_crop: "
            f"{bool(preprocessing['ref_crop_applied'])} "
            f"margins_lrtb={preprocessing['ref_crop_margins_lrtb']}"
        ),
        (
            "preprocessing_src_crop: "
            f"{bool(preprocessing['src_crop_applied'])} "
            f"margins_lrtb={preprocessing['src_crop_margins_lrtb']}"
        ),
        (
            "preprocessing_global_prealign: "
            f"enabled={bool(preprocessing['global_prealign_enabled'])}, "
            f"applied={bool(preprocessing['global_prealign_applied'])}, "
            f"scale={preprocessing['global_prealign_scale']}, "
            f"rot_deg={preprocessing['global_prealign_rotation_deg']}, "
            f"inlier_ratio={preprocessing['global_prealign_inlier_ratio']}, "
            f"median_err_px={preprocessing['global_prealign_median_err_px']}"
        ),
        "",
        "pred fields:",
    ]
    for key, meta in pred_shapes.items():
        if meta is None:
            summary_txt_lines.append(f"  - {key}: None")
        else:
            shape = meta["shape"]
            dtype = meta["dtype"]
            summary_txt_lines.append(f"  - {key}: shape={shape}, dtype={dtype}")

    summary_txt_lines.extend(
        [
            "",
            "warp implementation notes:",
            "  - Implemented using preds['warp_AB'] + torch.nn.functional.grid_sample (same direction as demo/demo_match.py).",
            "  - warp_AB was upsampled to reference size for full-resolution output.",
            "  - Masked warp visualization uses overlap_AB and in-bounds grid checks.",
            "  - Optional regularized warp blends dense warp with fallback in low-overlap regions.",
            "  - If overlap/precision were unavailable, this script still saves dense warps and sampled matches.",
            "",
            "artifacts:",
        ]
    )
    for p in output_files:
        summary_txt_lines.append(f"  - {p.resolve()}")

    summary_txt_path = args.outdir / "summary.txt"
    summary_txt_path.write_text("\n".join(summary_txt_lines) + "\n", encoding="utf-8")

    # --- Build numbered steps folder ---
    print("[INFO] Building numbered pipeline steps folder ...")
    steps_dir = args.outdir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension from the original reference image
    ref_ext = args.ref.suffix.lower() if args.ref.suffix else ".png"
    src_ext = args.src.suffix.lower() if args.src.suffix else ".png"

    step_entries: list[tuple[str, Path | None]] = [
        (f"1_original_bw{ref_ext}", args.ref),
        (f"2_ai_colorized{src_ext}", args.src),
        ("3_roma_dense_warp.png", warped_src_path),
        (
            "4_guided_filter_smooth.png",
            args.outdir / "warped_src_smooth.png" if gf_radius > 0 else None,
        ),
        (
            "5_final_color_transfer.png",
            args.outdir / "final_colorized.png" if not args.no_color_transfer else None,
        ),
    ]

    for step_name, source in step_entries:
        if source is None:
            continue
        source_path = Path(source) if not isinstance(source, Path) else source
        if source_path.exists():
            shutil.copy2(source_path, steps_dir / step_name)

    print(f"[INFO] Steps folder: {steps_dir.resolve()}")

    print("[INFO] Done.")
    print(f"[INFO] Summary: {summary_txt_path.resolve()}")
    print(f"[INFO] Main outputs: {args.outdir.resolve()}")
    print(f"[TIMING] Total runtime: {perf_counter() - run_start:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

