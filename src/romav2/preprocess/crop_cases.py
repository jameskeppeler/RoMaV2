"""Case-specific crop / transform handlers.

Each handler receives an image (BGR numpy array), extracted features,
and config, and returns the cropped/transformed image (or the original
unchanged) plus notes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .feature_extraction import (
    _border_mask,
    _order_quad,
    _sample_border_robust,
    _to_gray,
    find_edges_by_texture_scan,
    refine_crop_edges,
)
from .types import CropFeatures, PreprocConfig


# ── Strip-content validation ─────────────────────────────────────────

def _strip_has_content(gray: np.ndarray, threshold: float = 0.08) -> bool:
    """Return True if a gray strip contains distributed photo content (not just a border).

    Uses Canny edge density.  The threshold is calibrated to separate:
      - single colour-transition boundary rows (pink→white, etc.):  ~0.02-0.03
      - decorative / deckled / scalloped border notches:            ~0.04-0.07
      - real photo content (texture, objects, patterns):             0.08-0.20

    0.08 sits cleanly between the deckled-notch ceiling (~0.07) and the photo
    content floor (0.08), providing a safe margin on both sides.
    """
    if gray.size < 100:
        return False
    edges = cv2.Canny(gray, 30, 100)
    density = float(np.count_nonzero(edges)) / edges.size
    return density > threshold


def _validate_crop_bounds(
    gray: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    notes: list[str],
    *,
    edge_density_threshold: float = 0.08,
    min_strip_px: int = 8,
) -> tuple[int, int, int, int]:
    """Expand crop bounds if the strips being removed contain photo content.

    For each side, if the strip between the crop edge and the image edge
    has significant texture (edges), push the crop boundary outward to
    avoid cutting into photo content.
    """
    h, w = gray.shape[:2]
    adjusted = False

    # Top strip: gray[0:y1, x1:x2]
    if y1 > min_strip_px:
        strip = gray[0:y1, x1:x2]
        if strip.size > 0 and _strip_has_content(strip, edge_density_threshold):
            notes.append(f"Content found in top strip (0→{y1}), expanding crop to top")
            y1 = 0
            adjusted = True

    # Bottom strip: gray[y2:h, x1:x2]
    if (h - y2) > min_strip_px:
        strip = gray[y2:h, x1:x2]
        if strip.size > 0 and _strip_has_content(strip, edge_density_threshold):
            notes.append(f"Content found in bottom strip ({y2}→{h}), expanding crop to bottom")
            y2 = h
            adjusted = True

    # Left strip: gray[y1:y2, 0:x1]
    if x1 > min_strip_px:
        strip = gray[y1:y2, 0:x1]
        if strip.size > 0 and _strip_has_content(strip, edge_density_threshold):
            notes.append(f"Content found in left strip (0→{x1}), expanding crop to left")
            x1 = 0
            adjusted = True

    # Right strip: gray[y1:y2, x2:w]
    if (w - x2) > min_strip_px:
        strip = gray[y1:y2, x2:w]
        if strip.size > 0 and _strip_has_content(strip, edge_density_threshold):
            notes.append(f"Content found in right strip ({x2}→{w}), expanding crop to right")
            x2 = w
            adjusted = True

    if adjusted:
        notes.append(f"Crop bounds adjusted to ({x1},{y1})-({x2},{y2}) to preserve content")

    return x1, y1, x2, y2


# ── Uniform border (white or coloured) ───────────────────────────────

def crop_white_scan_border(
    img: np.ndarray,
    features: CropFeatures,
    cfg: PreprocConfig,
) -> tuple[np.ndarray, list[str]]:
    """Remove a uniform border (white scanner bed OR coloured mat/border).

    Strategy priority:
    A. Texture-scan: independently find each crop edge by scanning inward
       from each side and detecting where texture (local variance) begins.
       Most robust — immune to bright photo content near borders.
    B. Boundary contour: if texture scan fails and we have a good rectangular
       contour, crop to its bounding rect with gradient refinement.
    C. Colour-distance masking: fall back to masking pixels that differ from
       the sampled border colour.
    """
    notes: list[str] = []
    h, w = img.shape[:2]
    dim = min(h, w)
    gray = _to_gray(img)

    # --- Strategy A: per-side texture scan (preferred) ---
    # Run the texture scan at the working resolution (same as feature extraction)
    # to avoid a full-resolution variance-map computation on large scans.
    # The resulting coordinates are scaled back to original resolution.
    _tex_gray = gray
    _tex_inv = 1.0
    if cfg.max_working_res and min(h, w) > cfg.max_working_res:
        _ts = cfg.max_working_res / min(h, w)
        _tex_gray = cv2.resize(gray, (max(1, round(w * _ts)), max(1, round(h * _ts))),
                               interpolation=cv2.INTER_AREA)
        _tex_inv = min(h, w) / cfg.max_working_res

    tx1, ty1, tx2, ty2, tex_success, tex_notes = find_edges_by_texture_scan(_tex_gray)

    if tex_success and _tex_inv != 1.0:
        # Scale coordinates back; clamp to original image bounds
        tx1 = max(0, min(round(tx1 * _tex_inv), w - 1))
        ty1 = max(0, min(round(ty1 * _tex_inv), h - 1))
        tx2 = max(tx1 + 1, min(round(tx2 * _tex_inv), w))
        ty2 = max(ty1 + 1, min(round(ty2 * _tex_inv), h))

    notes.extend(tex_notes)

    if tex_success:
        # Validate: don't crop away strips that contain photo content
        tx1, ty1, tx2, ty2 = _validate_crop_bounds(gray, tx1, ty1, tx2, ty2, notes)

        # Sanity: crop must remove meaningful border (>= 1.5% on at least one side)
        min_strip = int(dim * 0.015)
        if tx1 >= min_strip or ty1 >= min_strip or (w - tx2) >= min_strip or (h - ty2) >= min_strip:
            cropped = img[ty1:ty2, tx1:tx2].copy()
            notes.append(
                f"Cropped via texture scan: ({tx1},{ty1})-({tx2},{ty2}) from {w}x{h}"
            )
            return cropped, notes
        else:
            notes.append("Texture scan found edges but border too narrow; trying contour fallback")

    # --- Strategy B: use the already-detected boundary contour ---
    contour = features.largest_contour
    if contour is not None and features.rectangle_likeness > 0.55:
        x, y, rw, rh = cv2.boundingRect(contour)
        margin = max(2, int(dim * 0.003))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + rw + margin)
        y2 = min(h, y + rh + margin)

        # Refine edges using gradient analysis (snaps to actual photo boundary)
        x1, y1, x2, y2, refine_notes = refine_crop_edges(gray, x1, y1, x2, y2)
        notes.extend(refine_notes)

        # Validate: don't crop away strips that contain photo content
        x1, y1, x2, y2 = _validate_crop_bounds(gray, x1, y1, x2, y2, notes)

        # Sanity: crop must remove meaningful border (>= 1.5% on at least one side)
        min_strip = int(dim * 0.015)
        if x1 >= min_strip or y1 >= min_strip or (w - x2) >= min_strip or (h - y2) >= min_strip:
            cropped = img[y1:y2, x1:x2].copy()
            notes.append(
                f"Cropped via boundary contour (strategy={features.contour_strategy}, "
                f"rect={features.rectangle_likeness:.2f}): "
                f"({x1},{y1})-({x2},{y2}) from {w}x{h}"
            )
            return cropped, notes

    # --- Strategy C: colour-distance masking with robust border sampling ---
    notes.append("Contour crop insufficient, falling back to colour-distance masking")
    band = max(4, int(dim * cfg.border_band_fraction))

    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        border_pixels = _sample_border_robust(lab, h, w, band)
        border_ref = np.median(border_pixels, axis=0)
        diff = np.linalg.norm(lab - border_ref, axis=2)

        # Multi-layer border: sample slightly deeper and mask both layers
        inner_band = min(band * 3, int(dim * 0.08))
        if inner_band > band + 4:
            inner_pixels = _sample_border_robust(lab, h, w, inner_band)
            inner_ref = np.median(inner_pixels, axis=0)
            layer_dist = float(np.linalg.norm(border_ref - inner_ref))
            if layer_dist > 25.0:
                diff2 = np.linalg.norm(lab - inner_ref, axis=2)
                diff = np.minimum(diff, diff2)
                notes.append(f"Multi-layer border detected (layer_dist={layer_dist:.1f})")

        q75 = np.percentile(border_pixels, 75, axis=0)
        q25 = np.percentile(border_pixels, 25, axis=0)
        border_iqr = float(np.mean(q75 - q25))
    else:
        gray_f = gray.astype(np.float32)
        border_pixels = _sample_border_robust(gray_f[:, :, np.newaxis], h, w, band).ravel()
        border_ref = float(np.median(border_pixels))
        diff = np.abs(gray_f - border_ref)
        border_iqr = float(np.percentile(border_pixels, 75) - np.percentile(border_pixels, 25))

    threshold = max(20.0, border_iqr * 2.5)
    content_mask = (diff > threshold).astype(np.uint8) * 255

    # Directional closing + square closing + open (same improved pipeline)
    k_dir = max(7, int(dim * 0.012) | 1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dir, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_dir))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_v, iterations=3)
    k_sq = max(5, int(dim * 0.006) | 1)
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sq, k_sq))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_sq, iterations=3)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel_sq, iterations=2)

    # Find the largest contour and use its convex hull for a clean bounding rect
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        notes.append("No content found after colour masking; returning original.")
        return img, notes

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    x, y, rw, rh = cv2.boundingRect(hull)

    margin = max(2, int(dim * 0.003))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + rw + margin)
    y2 = min(h, y + rh + margin)

    # Refine edges using gradient analysis
    x1, y1, x2, y2, refine_notes = refine_crop_edges(gray, x1, y1, x2, y2)
    notes.extend(refine_notes)

    # Validate: don't crop away strips that contain photo content
    x1, y1, x2, y2 = _validate_crop_bounds(gray, x1, y1, x2, y2, notes)

    min_strip = int(dim * 0.015)
    if x1 < min_strip and y1 < min_strip and (w - x2) < min_strip and (h - y2) < min_strip:
        notes.append("Border too narrow after colour masking; skipping crop.")
        return img, notes

    cropped = img[y1:y2, x1:x2].copy()
    notes.append(f"Cropped via colour-distance masking: ({x1},{y1})-({x2},{y2}) from {w}x{h}")
    return cropped, notes


# ── Rectangular border (non-white / coloured mat or frame) ───────────

def crop_rectangular_border(
    img: np.ndarray,
    features: CropFeatures,
    cfg: PreprocConfig,
) -> tuple[np.ndarray, list[str]]:
    """Crop to the largest rectangular contour found in the image.

    Works for photos with visible card mounts, coloured mats, etc.
    """
    notes: list[str] = []
    contour = features.largest_contour
    if contour is None:
        notes.append("No contour available for rectangular crop.")
        return img, notes

    h, w = img.shape[:2]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Use bounding rect (axis-aligned) if rotation is minimal OR the perspective
    # skew score (computed from the polygon approximation) says the image isn't
    # actually skewed.  Chamfered / cut corners on vintage photo mounts can push
    # cv2.minAreaRect to report angles of 5-15° even for perfectly straight scans,
    # so the angle threshold alone is not sufficient.
    angle = rect[2]
    low_skew = features.perspective_skew_score < cfg.perspective_skew_threshold
    if abs(angle) < 5 or abs(angle - 90) < 5 or abs(angle + 90) < 5 or low_skew:
        x, y, rw, rh = cv2.boundingRect(contour)
        # Small outward margin to avoid cutting right at the contour edge.
        margin = max(2, int(min(h, w) * 0.005))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + rw + margin)
        y2 = min(h, y + rh + margin)
        # NOTE: refine_crop_edges is intentionally NOT called here.
        # crop_rectangular_border crops to the contour boundary, which the
        # boundary-detection strategies already place accurately.  Applying
        # gradient refinement moves the boundary inward when a second inner
        # border layer (e.g. white/deckled paper inside a pink outer border)
        # creates a stronger apparent gradient than the actual outer edge.
        gray = _to_gray(img)
        # Validate: don't crop away strips that contain photo content
        x1, y1, x2, y2 = _validate_crop_bounds(gray, x1, y1, x2, y2, notes)
        cropped = img[y1:y2, x1:x2].copy()
        notes.append(f"Axis-aligned rectangular crop: ({x1},{y1})-({x2},{y2})")
        return cropped, notes

    # Rotated rectangle: warp to upright
    src_pts = _order_quad(box.astype(np.float32))
    rw_f, rh_f = rect[1]
    if rw_f < rh_f:
        rw_f, rh_f = rh_f, rw_f
    dst_pts = np.array(
        [[0, 0], [rw_f - 1, 0], [rw_f - 1, rh_f - 1], [0, rh_f - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (int(rw_f), int(rh_f)))
    notes.append(f"Rotated rectangular crop via perspective warp, angle={angle:.1f}")
    return cropped, notes


# ── Perspective correction (phone photo of a print) ──────────────────

def correct_perspective_and_crop(
    img: np.ndarray,
    features: CropFeatures,
    cfg: PreprocConfig,
) -> tuple[np.ndarray, list[str]]:
    """Detect a quadrilateral (photo edge) and warp it to a rectangle.

    Uses the quad stored in features.best_quad if available, otherwise
    attempts to find one from the largest contour.
    """
    notes: list[str] = []
    h, w = img.shape[:2]

    quad = features.best_quad
    if quad is None:
        # Try harder: find quad from contour with relaxed epsilon
        contour = features.largest_contour
        if contour is None:
            notes.append("No contour for perspective correction.")
            return img, notes
        peri = cv2.arcLength(contour, True)
        for eps in (0.05, 0.08, 0.10):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                quad = approx
                break
        if quad is None:
            notes.append("Could not approximate a quad from contour.")
            return img, notes

    pts = quad.reshape(4, 2).astype(np.float32)
    ordered = _order_quad(pts)

    # Compute output size from the quad
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])

    out_w = int(max(width_top, width_bot))
    out_h = int(max(height_left, height_right))
    if out_w < 10 or out_h < 10:
        notes.append("Quad too small for perspective correction.")
        return img, notes

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    notes.append(
        f"Perspective correction applied: quad warped to {out_w}x{out_h}, "
        f"skew={features.perspective_skew_score:.3f}"
    )
    return warped, notes


# ── Irregular / non-rectangular mask candidate ───────────────────────

def detect_irregular_mask_candidate(
    img: np.ndarray,
    features: CropFeatures,
    cfg: PreprocConfig,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Identify an irregular (oval, vignette, etc.) photo region.

    Returns (cropped_image, candidate_mask_or_None, notes).

    For now this uses ellipse fitting as a lightweight heuristic.
    A future SAM/rembg backend could replace the mask generation.
    """
    notes: list[str] = []
    h, w = img.shape[:2]

    # Try ellipse-based mask
    if features.best_ellipse is not None and features.ellipse_likeness > cfg.ellipse_fit_threshold:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, features.best_ellipse, 255, -1)
        # Crop to ellipse bounding rect
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, rw, rh = cv2.boundingRect(coords)
            margin = max(2, int(min(h, w) * 0.01))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + rw + margin)
            y2 = min(h, y + rh + margin)
            cropped = img[y1:y2, x1:x2].copy()
            mask_crop = mask[y1:y2, x1:x2].copy()
            notes.append(
                f"Ellipse-based irregular region detected "
                f"(likeness={features.ellipse_likeness:.2f}), cropped to ({x1},{y1})-({x2},{y2})"
            )
            return cropped, mask_crop, notes

    # Fallback: use largest contour bounding rect as a conservative estimate
    contour = features.largest_contour
    if contour is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        x, y, rw, rh = cv2.boundingRect(contour)
        margin = max(2, int(min(h, w) * 0.01))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + rw + margin)
        y2 = min(h, y + rh + margin)
        cropped = img[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2].copy()
        notes.append(
            "Irregular candidate: contour-based bounding crop (low confidence). "
            "Segmentation backend (SAM/rembg) could improve this."
        )
        return cropped, mask_crop, notes

    notes.append("No irregular region detected.")
    return img, None, notes
