from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
) -> int:
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
        x_ref, y_ref = float(kpts_ref[i, 0]), float(kpts_ref[i, 1])
        x_src, y_src = float(kpts_src[i, 0]), float(kpts_src[i, 1])
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


def main() -> int:
    args = parse_args()
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

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {args.outdir.resolve()}")

    ref_img = Image.open(args.ref).convert("RGB")
    src_img = Image.open(args.src).convert("RGB")
    ref_w, ref_h = ref_img.size
    src_w, src_h = src_img.size
    print(f"[INFO] Loaded images. ref={ref_w}x{ref_h}, src={src_w}x{src_h}")

    torch.set_float32_matmul_precision("highest")

    print(f"[INFO] Initializing RoMa v2 (setting={args.setting}, compile={args.compile}) ...")
    model = RoMaV2(RoMaV2.Cfg(setting=args.setting, compile=args.compile))

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

    print("[INFO] Running dense match (ref -> src) ...")
    preds = model.match(str(args.ref), str(args.src))
    print("[INFO] Match complete.")

    output_files: list[Path] = []
    pred_shapes: dict[str, dict[str, object] | None] = {}
    print("[INFO] Saving dense outputs (.npy) ...")
    for key, value in preds.items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            npy_path = args.outdir / f"{key}.npy"
            np.save(npy_path, array)
            output_files.append(npy_path)
            pred_shapes[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        else:
            pred_shapes[key] = None

    warp_ab = preds.get("warp_AB")
    if not isinstance(warp_ab, torch.Tensor):
        print("[ERROR] preds['warp_AB'] missing or invalid.", file=sys.stderr)
        return 1
    warp_ab = warp_ab[0]

    overlap_ab = preds.get("overlap_AB")
    overlap_small: torch.Tensor | None = None
    overlap_full: torch.Tensor | None = None
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
        overlap_full_np = overlap_full.detach().cpu().numpy()
        overlap_color_np = _overlap_to_color(overlap_full_np)

        overlap_color_path = args.outdir / "overlap_AB_color_refsize.png"
        Image.fromarray(overlap_color_np).save(overlap_color_path)
        output_files.append(overlap_color_path)

        ref_np = np.asarray(ref_img, dtype=np.float32)
        overlap_overlay = (
            0.65 * ref_np + 0.35 * overlap_color_np.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        overlap_overlay_path = args.outdir / "overlap_AB_overlay_on_ref.png"
        Image.fromarray(overlap_overlay).save(overlap_overlay_path)
        output_files.append(overlap_overlay_path)

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

        conf_mask_np = (confidence_mask.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
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

    # --- Guided-filter warp smoothing ---
    # Pipeline order: smooth RAW warp → THEN apply confidence mask (once).
    gf_radius = int(args.guided_filter_radius)
    warped_src_smooth: torch.Tensor | None = None
    if gf_radius > 0:
        print(
            f"[INFO] Applying guided-filter smoothing to warp field "
            f"(radius={gf_radius}, eps={args.guided_filter_eps:.1e}) ..."
        )
        ref_gray = np.asarray(ref_img.convert("L"), dtype=np.float32) / 255.0
        warp_np = warp_ab_full.detach().cpu().numpy()  # (H, W, 2)

        # Smooth the RAW warp (not the confidence-masked one)
        warp_smooth_np = np.stack(
            [
                _guided_filter(ref_gray, warp_np[..., 0], gf_radius, args.guided_filter_eps),
                _guided_filter(ref_gray, warp_np[..., 1], gf_radius, args.guided_filter_eps),
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
    else:
        print("[INFO] Guided-filter smoothing disabled (radius=0).")

    # --- Sampling correspondences (must happen before model is freed) ---
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
    )
    output_files.append(corr_vis_path)

    # Free RoMa model to reclaim VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Photoshop-style Color blend transfer ---
    final_colorized_tensor: torch.Tensor | None = None
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
            donor_a = _guided_filter(
                ref_gray,
                donor_lab[..., 1].astype(np.float32),
                chroma_radius,
                args.guided_filter_eps,
            )
            donor_b = _guided_filter(
                ref_gray,
                donor_lab[..., 2].astype(np.float32),
                chroma_radius,
                args.guided_filter_eps,
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

        final_colorized_tensor = (
            torch.from_numpy(final_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(roma_device)
        )
        print(
            "[INFO] Saved Photoshop-style Color blend result "
            f"(opacity={float(args.color_opacity):.2f}) -> final_colorized.png"
        )
    else:
        print("[INFO] Color transfer disabled (--no-color-transfer).")
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
    warped_regularized_tensor: torch.Tensor | None = None
    if overlap_full is not None and args.regularize_fallback != "none":
        tau = float(args.regularize_overlap_thresh)
        denom = max(1e-6, 1.0 - tau)
        alpha = ((overlap_full - tau) / denom).clamp(0.0, 1.0).unsqueeze(0)

        if args.regularize_fallback == "identity":
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
        elif args.regularize_fallback == "reference":
            fallback_img = ref_tensor
        else:
            fallback_img = warped_src

        warped_regularized = alpha * warped_src + (1.0 - alpha) * fallback_img
        warped_regularized_tensor = warped_regularized
        regularized_output_path = args.outdir / "warped_src_to_ref_regularized.png"
        _save_chw_tensor_png(warped_regularized, regularized_output_path)
        output_files.append(regularized_output_path)

        alpha_img = (alpha.squeeze(0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        alpha_path = args.outdir / "regularization_alpha.png"
        Image.fromarray(alpha_img).save(alpha_path)
        output_files.append(alpha_path)

    print("[INFO] Building before/after diagnostic images ...")
    resample = _resample_bicubic()
    src_resized_to_ref = src_img.resize((ref_w, ref_h), resample=resample)
    src_resized_tensor = _pil_to_tensor(src_resized_to_ref, device=roma_device)[0]

    before_after_triptych = torch.cat((ref_tensor, src_resized_tensor, warped_src), dim=2)
    before_after_triptych_path = args.outdir / "before_after_triptych.png"
    _save_chw_tensor_png(before_after_triptych, before_after_triptych_path)
    output_files.append(before_after_triptych_path)

    diff_before = (ref_tensor - src_resized_tensor).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
    diff_after = (ref_tensor - warped_src).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
    diff_panels = [diff_before, diff_after]
    if warped_regularized_tensor is not None:
        diff_after_reg = (ref_tensor - warped_regularized_tensor).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
        diff_panels.append(diff_after_reg)
    if final_colorized_tensor is not None:
        diff_after_color = (ref_tensor - final_colorized_tensor).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1)
        diff_panels.append(diff_after_color)
    diff_panel = torch.cat(diff_panels, dim=2)
    diff_panel_path = args.outdir / "difference_before_after.png"
    _save_chw_tensor_png(diff_panel, diff_panel_path)
    output_files.append(diff_panel_path)

    mae_before = float((ref_tensor - src_resized_tensor).abs().mean().item())
    mae_warped = float((ref_tensor - warped_src).abs().mean().item())
    mae_warped_regularized = (
        float((ref_tensor - warped_regularized_tensor).abs().mean().item())
        if warped_regularized_tensor is not None
        else None
    )
    mae_smooth_warped = (
        float((ref_tensor - warped_src_smooth).abs().mean().item())
        if warped_src_smooth is not None
        else None
    )
    mae_final_colorized = (
        float((ref_tensor - final_colorized_tensor).abs().mean().item())
        if final_colorized_tensor is not None
        else None
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
            "ref_size_wh": [ref_w, ref_h],
            "src_size_wh": [src_w, src_h],
        },
        "model": model_info,
        "regularization": {
            "fallback": args.regularize_fallback,
            "overlap_threshold": float(args.regularize_overlap_thresh),
            "regularized_output_written": regularized_output_path is not None,
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
        "metrics": {
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
        f"ref: {summary['inputs']['ref_path']} ({ref_w}x{ref_h})",
        f"src: {summary['inputs']['src_path']} ({src_w}x{src_h})",
        f"setting: {summary['model']['setting']}",
        f"compile: {summary['model']['compile']}",
        f"warp_map_hw: {h_map}x{w_map}",
        f"num_samples_requested: {args.num_samples}",
        f"num_samples_used: {num_samples}",
        f"num_drawn_correspondences: {drawn}",
        f"regularize_fallback: {args.regularize_fallback}",
        f"regularize_overlap_thresh: {args.regularize_overlap_thresh}",
        f"guided_filter_radius: {gf_radius}",
        f"guided_filter_eps: {args.guided_filter_eps}",
        f"color_transfer: {'enabled' if not args.no_color_transfer else 'disabled'}",
        f"color_opacity: {float(args.color_opacity):.3f}",
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

