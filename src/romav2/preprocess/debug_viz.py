"""Debug visualisation utilities for the crop-routing pipeline.

All functions return BGR numpy arrays suitable for display or saving.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import CropFeatures, CropResult


def draw_contour_overlay(img: np.ndarray, features: CropFeatures) -> np.ndarray:
    """Draw the largest contour and its polygon approximation on the image."""
    vis = img.copy()
    if features.largest_contour is not None:
        cv2.drawContours(vis, [features.largest_contour], -1, (0, 255, 0), 2)
    if features.approx_polygon is not None:
        cv2.drawContours(vis, [features.approx_polygon], -1, (0, 200, 255), 2)
    return vis


def draw_quad_overlay(img: np.ndarray, features: CropFeatures) -> np.ndarray:
    """Draw the detected quadrilateral (if any) for perspective correction."""
    vis = img.copy()
    if features.best_quad is not None:
        pts = features.best_quad.reshape(-1, 2).astype(np.int32)
        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(vis, p1, p2, (255, 0, 255), 3)
            cv2.circle(vis, p1, 8, (0, 0, 255), -1)
    return vis


def draw_ellipse_overlay(img: np.ndarray, features: CropFeatures) -> np.ndarray:
    """Draw the fitted ellipse (if any)."""
    vis = img.copy()
    if features.best_ellipse is not None:
        cv2.ellipse(vis, features.best_ellipse, (255, 255, 0), 2)
    return vis


def draw_border_heatmap(img: np.ndarray, features: CropFeatures, band_frac: float = 0.05) -> np.ndarray:
    """Show the outer band highlighted as a translucent overlay."""
    vis = img.copy()
    h, w = vis.shape[:2]
    band = max(1, int(min(h, w) * band_frac))
    overlay = vis.copy()
    # Tint the outer band
    overlay[:band, :] = (0, 0, 200)        # top
    overlay[h - band:, :] = (0, 0, 200)    # bottom
    overlay[band:h - band, :band] = (0, 0, 200)        # left
    overlay[band:h - band, w - band:] = (0, 0, 200)    # right
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


def draw_crop_preview(result: CropResult) -> np.ndarray:
    """Side-by-side: original (left) and cropped result (right).

    Both images are resized to the same height for visual comparison.
    """
    orig = result.original
    crop = result.image
    target_h = min(orig.shape[0], 800)

    def _resize(im: np.ndarray, th: int) -> np.ndarray:
        h, w = im.shape[:2]
        if h == th:
            return im
        scale = th / h
        return cv2.resize(im, (int(w * scale), th), interpolation=cv2.INTER_AREA)

    left = _resize(orig, target_h)
    right = _resize(crop, target_h)

    # Pad the shorter one on the right
    max_w = max(left.shape[1], right.shape[1])

    def _pad_w(im: np.ndarray, tw: int) -> np.ndarray:
        if im.shape[1] >= tw:
            return im
        pad = np.zeros((im.shape[0], tw - im.shape[1], 3), dtype=im.dtype)
        return np.hstack([im, pad])

    left = _pad_w(left, max_w)
    right = _pad_w(right, max_w)

    # Separator
    sep = np.full((target_h, 4, 3), 128, dtype=np.uint8)
    return np.hstack([left, sep, right])


def build_debug_panel(result: CropResult) -> np.ndarray:
    """Build a composite debug panel with key overlays arranged in a grid.

    Layout (2x2 when all available):
        [ border heatmap  |  contour overlay ]
        [ quad/ellipse    |  crop preview    ]
    """
    panels: list[np.ndarray] = []
    orig = result.original
    feats = result.features

    panels.append(draw_border_heatmap(orig, feats))
    panels.append(draw_contour_overlay(orig, feats))

    # Third panel: quad or ellipse depending on case
    if feats.best_quad is not None:
        panels.append(draw_quad_overlay(orig, feats))
    elif feats.best_ellipse is not None:
        panels.append(draw_ellipse_overlay(orig, feats))
    else:
        panels.append(orig.copy())

    # Fourth panel: crop result
    if result.was_cropped:
        panels.append(result.image)
    else:
        panels.append(orig.copy())

    # Resize all to same size
    target_h = min(orig.shape[0], 500)

    def _fit(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        scale = target_h / h
        return cv2.resize(im, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

    panels = [_fit(p) for p in panels]

    # Make all same width (pad)
    max_w = max(p.shape[1] for p in panels)

    def _pad(im: np.ndarray) -> np.ndarray:
        if im.shape[1] >= max_w:
            return im
        pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=im.dtype)
        return np.hstack([im, pad])

    panels = [_pad(p) for p in panels]

    row1 = np.hstack([panels[0], panels[1]])
    row2 = np.hstack([panels[2], panels[3]])
    return np.vstack([row1, row2])


def save_debug_outputs(result: CropResult, outdir: Path) -> list[Path]:
    """Save individual debug images to *outdir*. Returns list of saved paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _save(name: str, img: np.ndarray) -> None:
        p = outdir / name
        cv2.imwrite(str(p), img)
        saved.append(p)

    _save("prep_original.png", result.original)

    feats = result.features
    _save("prep_border_heatmap.png", draw_border_heatmap(result.original, feats))
    _save("prep_contour_overlay.png", draw_contour_overlay(result.original, feats))

    if feats.best_quad is not None:
        _save("prep_quad_overlay.png", draw_quad_overlay(result.original, feats))
    if feats.best_ellipse is not None:
        _save("prep_ellipse_overlay.png", draw_ellipse_overlay(result.original, feats))

    if result.was_cropped:
        _save("prep_crop_result.png", result.image)
        _save("prep_side_by_side.png", draw_crop_preview(result))

    _save("prep_debug_panel.png", build_debug_panel(result))

    return saved
