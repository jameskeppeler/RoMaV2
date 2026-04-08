"""Image feature extraction for the crop-routing pipeline.

All functions operate on numpy arrays (BGR or grayscale) and return
scalar features or annotated structures from ``types.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import CropFeatures, PreprocConfig


# ── Vectorised scan-line onset detector ──────────────────────────────

def _find_onset_per_scanline(
    var_strip: np.ndarray,
    threshold: float,
    run_required: int,
) -> list[int]:
    """Return the onset position for each scan-line that has a qualifying run.

    A qualifying run is ``run_required`` consecutive positions where the local
    variance exceeds ``threshold``.  The returned value for each scan-line is
    the *start* index of the first such run (i.e. the distance from the image
    edge to the first textured pixel).

    Fully vectorised with NumPy; replaces the previous nested Python loop.

    Parameters
    ----------
    var_strip : (N, D) float32 array
        Local-variance values for N scan-lines, each of depth D.
    threshold : float
    run_required : int  (>= 1)

    Returns
    -------
    list[int]
        One entry per scan-line that detected a texture onset.
    """
    N, D = var_strip.shape
    if D == 0 or N == 0:
        return []

    above = var_strip > threshold          # (N, D) bool

    if run_required <= 1:
        # Simple: first True in each row
        first = np.argmax(above, axis=1)   # (N,) — 0 when all-False too
        valid = above[np.arange(N), first]
        return [int(first[i]) for i in range(N) if valid[i]]

    # Running window-sum: ws[i, j] == run_required  <=>  above[i, j:j+run_required] all True
    W = D - run_required + 1
    if W <= 0:
        return []

    cum = np.cumsum(above.view(np.uint8), axis=1, dtype=np.int16)  # (N, D)
    ws = cum[:, run_required - 1:].copy()   # (N, W)  — ws[:,j] = cum[:,j+rr-1]
    ws[:, 1:] -= cum[:, :W - 1]            # subtract prior prefix: ws[:,j] = sum(above[:,j:j+rr])

    full_run = ws >= run_required           # (N, W)
    first_win = np.argmax(full_run, axis=1) # (N,) — index of first complete run
    has_run = full_run[np.arange(N), first_win]

    return [int(first_win[i]) for i in range(N) if has_run[i]]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ── Gradient-based edge refinement ────────────────────────────────────

def refine_crop_edges(
    gray: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *,
    search_inward: int | None = None,
    search_outward: int | None = None,
) -> tuple[int, int, int, int, list[str]]:
    """Refine crop boundaries by finding the strongest gradient transitions.

    For each edge of the bounding rect, scan perpendicular strips in the
    middle portion of the edge (avoiding corners where staining lives) to
    find where the strongest brightness transition occurs.  Snaps the
    boundary to that transition line.

    Returns (x1, y1, x2, y2, notes).
    """
    h, w = gray.shape[:2]
    dim = min(h, w)
    notes: list[str] = []

    if search_inward is None:
        search_inward = max(8, int(dim * 0.04))
    if search_outward is None:
        search_outward = max(8, int(dim * 0.03))

    gray_f = gray.astype(np.float32)

    # Skip 15 % on each end of every edge to avoid corner staining
    edge_skip = int(dim * 0.15)

    def _find_edge_transition(profile: np.ndarray, min_gradient: float = 8.0) -> int | None:
        """Find the index of the strongest transition in a 1-D profile."""
        if len(profile) < 5:
            return None
        # Smooth to suppress noise but keep the real edge
        k = max(3, len(profile) // 8)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k) / k
        smoothed = np.convolve(profile, kernel, mode="same")
        grad = np.abs(np.diff(smoothed))
        if len(grad) == 0:
            return None
        peak = int(np.argmax(grad))
        if grad[peak] < min_gradient:
            return None
        return peak

    orig = (x1, y1, x2, y2)

    # --- Top edge: scan from y1-out to y1+in ---
    s_top = max(0, y1 - search_outward)
    s_bot = min(h, y1 + search_inward)
    ml = x1 + edge_skip
    mr = x2 - edge_skip
    if s_bot > s_top + 4 and mr > ml + 20:
        strip = gray_f[s_top:s_bot, ml:mr]
        profile = np.mean(strip, axis=1)
        idx = _find_edge_transition(profile)
        if idx is not None:
            # +1: snap to the first content row, just past the gradient peak
            y1 = s_top + idx + 1

    # --- Bottom edge ---
    s_top = max(0, y2 - search_inward)
    s_bot = min(h, y2 + search_outward)
    if s_bot > s_top + 4 and mr > ml + 20:
        strip = gray_f[s_top:s_bot, ml:mr]
        profile = np.mean(strip, axis=1)
        idx = _find_edge_transition(profile)
        if idx is not None:
            # No +1: y2 is an exclusive end, so the gradient peak row is the last border row
            y2 = s_top + idx

    # --- Left edge ---
    s_left = max(0, x1 - search_outward)
    s_right = min(w, x1 + search_inward)
    mt = y1 + edge_skip
    mb = y2 - edge_skip
    if s_right > s_left + 4 and mb > mt + 20:
        strip = gray_f[mt:mb, s_left:s_right]
        profile = np.mean(strip, axis=0)
        idx = _find_edge_transition(profile)
        if idx is not None:
            # +1: snap to the first content column, just past the gradient peak
            x1 = s_left + idx + 1

    # --- Right edge ---
    s_left = max(0, x2 - search_inward)
    s_right = min(w, x2 + search_outward)
    if s_right > s_left + 4 and mb > mt + 20:
        strip = gray_f[mt:mb, s_left:s_right]
        profile = np.mean(strip, axis=0)
        idx = _find_edge_transition(profile)
        if idx is not None:
            # No +1: x2 is an exclusive end, so the gradient peak column is the last border column
            x2 = s_left + idx

    # Safety: don't let refinement shrink any dimension by more than 15%
    orig_w = orig[2] - orig[0]
    orig_h = orig[3] - orig[1]
    new_w = x2 - x1
    new_h = y2 - y1
    if orig_w > 0 and new_w < orig_w * 0.85:
        x1, x2 = orig[0], orig[2]  # revert horizontal
    if orig_h > 0 and new_h < orig_h * 0.85:
        y1, y2 = orig[1], orig[3]  # revert vertical

    if (x1, y1, x2, y2) != orig:
        notes.append(
            f"Edge refinement adjusted bounds: "
            f"({orig[0]},{orig[1]})-({orig[2]},{orig[3]}) -> ({x1},{y1})-({x2},{y2})"
        )

    return x1, y1, x2, y2, notes


def find_edges_by_texture_scan(
    gray: np.ndarray,
    *,
    window_size: int | None = None,
    variance_threshold: float | None = None,
    num_scanlines: int = 40,
    scan_depth_fraction: float = 0.35,
    min_consensus_fraction: float = 0.3,
) -> tuple[int, int, int, int, bool, list[str]]:
    """Find crop edges by scanning inward from each side, detecting texture onset.

    Instead of finding a contour, this independently locates each crop edge by
    casting perpendicular scan lines from the image border inward.  Along each
    scan line, local variance is computed in a sliding window.  The first
    position where variance exceeds the threshold marks the photo boundary.
    The median of all scan-line detections gives a robust crop coordinate per
    side that is immune to bright photo content near the border.

    Returns (x1, y1, x2, y2, success, notes).
    ``success`` is True when enough scan lines agreed on all 4 sides.
    """
    h, w = gray.shape[:2]
    dim = min(h, w)
    notes: list[str] = []

    if window_size is None:
        window_size = max(7, int(dim * 0.015) | 1)
    if variance_threshold is None:
        # Adaptive: compute the median variance of the inner 50% of the image
        # to calibrate what "textured" means for this particular photo.
        center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if center.size > 100:
            # Use local variance of the center as a reference
            center_f = center.astype(np.float32)
            center_mean = cv2.blur(center_f, (window_size, window_size))
            center_sq_mean = cv2.blur(center_f ** 2, (window_size, window_size))
            center_var = center_sq_mean - center_mean ** 2
            median_var = float(np.median(center_var[center_var > 0])) if np.any(center_var > 0) else 50.0
            # Threshold: a fraction of the center's typical variance
            variance_threshold = max(15.0, median_var * 0.15)
        else:
            variance_threshold = 30.0

    max_scan = int(dim * scan_depth_fraction)
    gray_f = gray.astype(np.float32)

    # Precompute local variance map using box filter: Var = E[X^2] - E[X]^2
    local_mean = cv2.blur(gray_f, (window_size, window_size))
    local_sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    var_map = local_sq_mean - local_mean ** 2
    var_map = np.maximum(var_map, 0.0)  # clamp numerical noise

    # Skip corners: only scan the middle 60% of each edge
    corner_skip = int(dim * 0.20)

    edges: dict[str, int | None] = {}
    min_votes = int(num_scanlines * min_consensus_fraction)
    run_required = max(2, window_size // 3)

    # --- Top edge: scan downward from row 0 ---
    # scan_x holds column (x) positions evenly spaced along the top/bottom edges
    scan_x = np.linspace(corner_skip, w - corner_skip, num_scanlines, dtype=int)
    scan_x = np.clip(scan_x, 0, w - 1)
    depth = min(max_scan, h)
    # Build (num_scanlines, depth) strip matrix in one advanced-index gather — no Python loop
    strips_top = var_map[:depth, scan_x].T.astype(np.float32)           # (N, depth)
    onsets = _find_onset_per_scanline(strips_top, variance_threshold, run_required)
    if len(onsets) >= min_votes:
        edges["top"] = int(np.median(onsets))
        notes.append(f"Top edge: {len(onsets)}/{num_scanlines} scanlines, median={edges['top']}")
    else:
        edges["top"] = None
        notes.append(f"Top edge: insufficient consensus ({len(onsets)}/{num_scanlines})")

    # --- Bottom edge: scan upward from row h-1 ---
    strips_bot = var_map[h - depth:h, scan_x].T[:, ::-1].astype(np.float32)  # flipped
    onsets = _find_onset_per_scanline(strips_bot, variance_threshold, run_required)
    if len(onsets) >= min_votes:
        edges["bottom"] = h - int(np.median(onsets))
        notes.append(f"Bottom edge: {len(onsets)}/{num_scanlines} scanlines, median at row {edges['bottom']}")
    else:
        edges["bottom"] = None
        notes.append(f"Bottom edge: insufficient consensus ({len(onsets)}/{num_scanlines})")

    # --- Left edge: scan rightward from col 0 ---
    scan_y = np.linspace(corner_skip, h - corner_skip, num_scanlines, dtype=int)
    scan_y = np.clip(scan_y, 0, h - 1)
    depth_h = min(max_scan, w)
    strips_left = var_map[scan_y, :depth_h].astype(np.float32)          # (N, depth_h)
    onsets = _find_onset_per_scanline(strips_left, variance_threshold, run_required)
    if len(onsets) >= min_votes:
        edges["left"] = int(np.median(onsets))
        notes.append(f"Left edge: {len(onsets)}/{num_scanlines} scanlines, median={edges['left']}")
    else:
        edges["left"] = None
        notes.append(f"Left edge: insufficient consensus ({len(onsets)}/{num_scanlines})")

    # --- Right edge: scan leftward from col w-1 ---
    strips_right = var_map[scan_y, w - depth_h:w][:, ::-1].astype(np.float32)  # flipped
    onsets = _find_onset_per_scanline(strips_right, variance_threshold, run_required)
    if len(onsets) >= min_votes:
        edges["right"] = w - int(np.median(onsets))
        notes.append(f"Right edge: {len(onsets)}/{num_scanlines} scanlines, median at col {edges['right']}")
    else:
        edges["right"] = None
        notes.append(f"Right edge: insufficient consensus ({len(onsets)}/{num_scanlines})")

    # Build final crop rect — fall back to image edge where scan failed
    x1 = edges["left"] if edges["left"] is not None else 0
    y1 = edges["top"] if edges["top"] is not None else 0
    x2 = edges["right"] if edges["right"] is not None else w
    y2 = edges["bottom"] if edges["bottom"] is not None else h

    # Clamp
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    success = all(v is not None for v in edges.values())
    notes.append(
        f"Texture scan result: ({x1},{y1})-({x2},{y2}) from {w}x{h}, "
        f"success={success}, var_thresh={variance_threshold:.1f}, win={window_size}"
    )

    return x1, y1, x2, y2, success, notes


def _border_mask(h: int, w: int, band: int) -> np.ndarray:
    """Return a boolean mask that is True in the outer band, False in center."""
    mask = np.ones((h, w), dtype=bool)
    mask[band:h - band, band:w - band] = False
    return mask


def _center_mask(h: int, w: int, band: int) -> np.ndarray:
    """Return a boolean mask that is True in the centre rectangle."""
    mask = np.zeros((h, w), dtype=bool)
    mask[band:h - band, band:w - band] = True
    return mask


# ── Border brightness / variance ──────────────────────────────────────

def compute_border_stats(
    gray: np.ndarray,
    cfg: PreprocConfig,
) -> tuple[float, float, float]:
    """Return (mean_brightness, std_brightness, variance) of the outer band."""
    h, w = gray.shape[:2]
    band = max(1, int(min(h, w) * cfg.border_band_fraction))
    bm = _border_mask(h, w, band)
    pixels = gray[bm].astype(np.float64)
    if pixels.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(pixels))
    std = float(np.std(pixels))
    return mean, std, std * std


def compute_border_color_uniformity(
    img: np.ndarray,
    cfg: PreprocConfig,
    *,
    _lab: np.ndarray | None = None,
    _border_px: np.ndarray | None = None,
) -> tuple[float, float]:
    """Measure how uniform the outer band is in colour space.

    Returns (uniformity_score 0-1, border_vs_center_color_distance).
    High uniformity + high distance = likely a coloured border/mat.
    Samples from edge midpoints to avoid corner staining.

    Parameters
    ----------
    _lab : optional pre-computed Lab image (float32).
    _border_px : optional pre-computed border pixel sample (float32).
    """
    if img.ndim != 3:
        return 0.0, 0.0
    h, w = img.shape[:2]
    band = max(1, int(min(h, w) * cfg.border_band_fraction))
    cm = _center_mask(h, w, band)

    lab = _lab if _lab is not None else cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    border_pixels = _border_px if _border_px is not None else _sample_border_robust(lab, h, w, band)
    center_pixels = lab[cm]

    if border_pixels.size == 0 or center_pixels.size == 0:
        return 0.0, 0.0

    # Uniformity: use IQR-based spread (robust to outliers like stains)
    # Cast to float32 for uniform precision regardless of how _border_px was created
    bp = border_pixels.astype(np.float32)
    cp = center_pixels.astype(np.float32)
    q75 = np.percentile(bp, 75, axis=0)
    q25 = np.percentile(bp, 25, axis=0)
    border_iqr = float(np.mean(q75 - q25))
    # Normalise: IQR of 0 => 1.0, IQR of 40+ => ~0
    uniformity = float(np.clip(1.0 - border_iqr / 40.0, 0.0, 1.0))

    # Color distance between border and center medians
    border_median = np.median(bp, axis=0)
    center_median = np.median(cp, axis=0)
    dist = float(np.linalg.norm(border_median - center_median))
    dist_norm = float(np.clip(dist / 40.0, 0.0, 1.0))

    return uniformity, dist_norm


# ── Edge density ──────────────────────────────────────────────────────

def compute_edge_densities(
    gray: np.ndarray,
    cfg: PreprocConfig,
) -> tuple[float, float, float]:
    """Return (border_edge_density, center_edge_density, ratio).

    Edge density = fraction of pixels above the Canny threshold.
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape[:2]
    band = max(1, int(min(h, w) * cfg.border_band_fraction))
    bm = _border_mask(h, w, band)
    cm = _center_mask(h, w, band)

    border_count = float(np.count_nonzero(edges[bm]))
    center_count = float(np.count_nonzero(edges[cm]))
    border_area = float(np.count_nonzero(bm))
    center_area = float(np.count_nonzero(cm))

    bd = border_count / border_area if border_area > 0 else 0.0
    cd = center_count / center_area if center_area > 0 else 0.0
    ratio = bd / cd if cd > 1e-9 else 0.0
    return bd, cd, ratio


# ── Contour analysis ─────────────────────────────────────────────────

def _find_contour_by_otsu(
    gray: np.ndarray,
    *,
    _blurred: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Strategy 1: inverted Otsu (finds darkest blob). Good for white borders."""
    blurred = _blurred if _blurred is not None else cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    img_area = gray.shape[0] * gray.shape[1]
    return largest, cv2.contourArea(largest) / img_area


def _sample_border_robust(
    pixels: np.ndarray,
    h: int, w: int,
    band: int,
) -> np.ndarray:
    """Sample border colour from the middle portion of each edge, avoiding corners.

    Corners are where aging, staining, and yellowing concentrate on old photos.
    By sampling only the central 60% of each edge strip we get a much cleaner
    estimate of the true border colour.

    *pixels* is the full image in the working colour space (Lab or gray),
    shape (H, W) or (H, W, C).
    Returns 1-D array of sampled pixels (N,) or (N, C).
    """
    # Avoid the corner quadrants: skip 20% on each end of every edge
    margin = int(min(h, w) * 0.20)
    samples: list[np.ndarray] = []

    # Top edge (middle strip)
    samples.append(pixels[:band, margin:w - margin].reshape(-1, *pixels.shape[2:]))
    # Bottom edge
    samples.append(pixels[h - band:, margin:w - margin].reshape(-1, *pixels.shape[2:]))
    # Left edge
    samples.append(pixels[margin:h - margin, :band].reshape(-1, *pixels.shape[2:]))
    # Right edge
    samples.append(pixels[margin:h - margin, w - band:].reshape(-1, *pixels.shape[2:]))

    return np.concatenate(samples, axis=0)


def _find_contour_by_border_color(
    img: np.ndarray,
    gray: np.ndarray,
    cfg: PreprocConfig,
    *,
    _lab: np.ndarray | None = None,
    _border_px: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Strategy 2: sample the border colour (avoiding corners), mask 'not-border'
    pixels, find the largest contour in the result.

    This finds the photo rectangle inside a coloured or white border.
    Robust to corner staining/yellowing on aged photographs.

    Handles multi-layer borders (e.g. pink outer + white inner) by detecting
    if the initial border sample contains two distinct colour clusters and
    masking both layers.
    """
    h, w = gray.shape[:2]
    band = max(4, int(min(h, w) * cfg.border_band_fraction))
    dim = min(h, w)

    if img.ndim == 3:
        # Reuse cached Lab image and border pixels when available
        lab = _lab if _lab is not None else cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        border_pixels = _border_px if _border_px is not None else _sample_border_robust(lab, h, w, band)
        border_ref = np.median(border_pixels, axis=0)
        diff = np.linalg.norm(lab - border_ref, axis=2)

        # --- Multi-layer border detection ---
        # Check if there's a NARROW second border colour (e.g. white inner
        # border just inside a pink outer border).  Only sample slightly
        # deeper than the outer band — going too deep picks up photo content.
        inner_band = min(band * 3, int(dim * 0.08))
        if inner_band > band + 4:
            inner_ring_pixels = _sample_border_robust(lab, h, w, inner_band)
            inner_ref = np.median(inner_ring_pixels, axis=0)
            layer_dist = float(np.linalg.norm(border_ref - inner_ref))

            # Require a LARGE colour difference — inner border must be
            # clearly a different colour, not just slightly different content
            if layer_dist > 25.0:
                diff2 = np.linalg.norm(lab - inner_ref, axis=2)
                diff = np.minimum(diff, diff2)
    else:
        gray_f = gray.astype(np.float32)
        border_pixels = _sample_border_robust(gray_f[:, :, np.newaxis], h, w, band).ravel()
        border_ref = float(np.median(border_pixels))
        diff = np.abs(gray_f - border_ref)

    # Adaptive threshold based on the sampled border's spread (IQR-based)
    if img.ndim == 3:
        q75 = np.percentile(border_pixels, 75, axis=0)
        q25 = np.percentile(border_pixels, 25, axis=0)
        border_iqr = float(np.mean(q75 - q25))
    else:
        border_iqr = float(np.percentile(border_pixels, 75) - np.percentile(border_pixels, 25))
    threshold = max(20.0, border_iqr * 2.5)

    content_mask = (diff > threshold).astype(np.uint8) * 255

    # --- Improved morphological pipeline ---
    # 1. Directional closing: close horizontal and vertical gaps separately
    #    to preserve the rectangular structure while bridging corner gaps
    k_h = max(7, int(dim * 0.012) | 1)
    k_v = max(7, int(dim * 0.012) | 1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_v))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_h, iterations=3)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_v, iterations=3)

    # 2. Square closing to fill any remaining interior gaps
    k_sq = max(5, int(dim * 0.006) | 1)
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sq, k_sq))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel_sq, iterations=3)

    # 3. Open to remove small noise blobs
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel_sq, iterations=2)

    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    # Pick the largest contour, then use its convex hull for a cleaner boundary
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    img_area = h * w
    ratio = cv2.contourArea(hull) / img_area
    return hull, ratio


def _find_boundary_by_hough_lines(
    gray: np.ndarray,
    cfg: PreprocConfig,
    *,
    _blurred: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Strategy 4: Hough line detection to find rectangular photo boundary.

    Finds strong straight horizontal/vertical lines and assembles them into
    the tightest rectangular boundary.  Excellent for scanned photos with
    clear rectilinear borders.
    """
    h, w = gray.shape[:2]
    dim = min(h, w)
    blurred = _blurred if _blurred is not None else cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    min_line_len = int(dim * 0.15)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(50, int(dim * 0.05)),
        minLineLength=min_line_len,
        maxLineGap=max(10, int(dim * 0.02)),
    )
    if lines is None or len(lines) < 4:
        return None, 0.0

    h_lines: list[tuple[float, float, float, float]] = []   # (avg_y, x_min, x_max, length)
    v_lines: list[tuple[float, float, float, float]] = []   # (avg_x, y_min, y_max, length)

    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        dx = abs(lx2 - lx1)
        dy = abs(ly2 - ly1)
        length = np.sqrt(dx * dx + dy * dy)
        if dx > 3 * dy:  # horizontal
            h_lines.append(((ly1 + ly2) / 2, min(lx1, lx2), max(lx1, lx2), length))
        elif dy > 3 * dx:  # vertical
            v_lines.append(((lx1 + lx2) / 2, min(ly1, ly2), max(ly1, ly2), length))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None, 0.0

    # Cluster lines and pick the best top/bottom/left/right
    min_len = dim * 0.10

    def _best_line(candidates, region_lo, region_hi):
        """Pick the strongest (longest) line in the given region."""
        valid = [(c, c[3]) for c in candidates if region_lo <= c[0] <= region_hi and c[3] >= min_len]
        if not valid:
            return None
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[0][0]

    top_line = _best_line(h_lines, 0, h * 0.40)
    bot_line = _best_line(h_lines, h * 0.60, h)
    left_line = _best_line(v_lines, 0, w * 0.40)
    right_line = _best_line(v_lines, w * 0.60, w)

    if any(l is None for l in (top_line, bot_line, left_line, right_line)):
        return None, 0.0

    ty = int(top_line[0])
    by = int(bot_line[0])
    lx = int(left_line[0])
    rx = int(right_line[0])

    if by - ty < dim * 0.3 or rx - lx < dim * 0.3:
        return None, 0.0

    rect_contour = np.array([
        [[lx, ty]], [[rx, ty]], [[rx, by]], [[lx, by]],
    ], dtype=np.int32)

    area = (rx - lx) * (by - ty)
    ratio = area / (h * w)
    return rect_contour, ratio


def _find_contour_by_edges(
    gray: np.ndarray,
    *,
    _blurred: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Strategy 3: Canny edges -> find largest rectangular-ish contour.

    Good for finding sharp photo boundaries.
    """
    blurred = _blurred if _blurred is not None else cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # Dilate to close gaps in the edge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    img_area = gray.shape[0] * gray.shape[1]
    # Filter for large, roughly rectangular contours
    best = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / img_area
        if ratio < 0.15:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Prefer 4-vertex approximations (rectangles); matches the bonus used in find_photo_boundary
        vert_bonus = 1.4 if len(approx) == 4 else 1.0
        score = ratio * vert_bonus
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        # Fall back to largest
        best = max(contours, key=cv2.contourArea)
    return best, cv2.contourArea(best) / img_area


def find_photo_boundary(
    img: np.ndarray,
    gray: np.ndarray,
    cfg: PreprocConfig,
    *,
    _gray_blurred: np.ndarray | None = None,
    _lab: np.ndarray | None = None,
    _border_px: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, str]:
    """Try multiple strategies to find the photo boundary contour.

    Returns (best_contour, area_ratio, strategy_name).
    Picks the candidate that is most rectangular and covers a good area.

    Parameters
    ----------
    _gray_blurred : optional pre-computed GaussianBlur(gray, 5, 5) — shared
        across the three blur-dependent strategies to avoid triple recomputation.
    _lab : optional pre-computed Lab image (float32) — shared with
        ``_find_contour_by_border_color`` to avoid repeated colour conversion.
    _border_px : optional pre-computed border pixel sample in Lab space.
    """
    # Shared blurred-gray: computed once, reused by Otsu / edges / Hough
    blurred = _gray_blurred if _gray_blurred is not None else cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: list[tuple[np.ndarray | None, float, str]] = []

    # Strategy 1: border-colour masking (best for coloured/white borders)
    c1, r1 = _find_contour_by_border_color(img, gray, cfg, _lab=_lab, _border_px=_border_px)
    candidates.append((c1, r1, "border_color"))

    # Short-circuit: if strategy 1 already found a high-quality 4-vertex rectangular
    # contour, the remaining strategies are very unlikely to improve on it.
    # Compute the same scoring formula used in the final selection loop below.
    if c1 is not None and r1 >= cfg.min_contour_area_ratio:
        _peri1 = cv2.arcLength(c1, True)
        _approx1 = cv2.approxPolyDP(c1, 0.02 * _peri1, True)
        if len(_approx1) == 4:
            _rect1 = cv2.minAreaRect(c1)
            _ra1 = _rect1[1][0] * _rect1[1][1]
            _fill1 = cv2.contourArea(c1) / _ra1 if _ra1 > 1 else 0.0
            _aq1 = r1 if r1 < 0.95 else r1 * 0.3
            if _fill1 * 1.4 * _aq1 > 1.0:   # near-perfect rectangular result
                return c1, r1, "border_color"

    # Strategy 2: Otsu (legacy, good for high-contrast white borders)
    c2, r2 = _find_contour_by_otsu(gray, _blurred=blurred)
    candidates.append((c2, r2, "otsu"))

    # Strategy 3: edge-based (good for sharp boundaries)
    c3, r3 = _find_contour_by_edges(gray, _blurred=blurred)
    candidates.append((c3, r3, "edge"))

    # Strategy 4: Hough lines (strong rectilinear boundaries)
    c4, r4 = _find_boundary_by_hough_lines(gray, cfg, _blurred=blurred)
    candidates.append((c4, r4, "hough_lines"))

    # Score each candidate: prefer rectangular + good area coverage
    best_contour = None
    best_area_ratio = 0.0
    best_strategy = "none"
    best_total_score = -1.0

    for contour, area_ratio, strategy in candidates:
        if contour is None or area_ratio < cfg.min_contour_area_ratio:
            continue

        # How rectangular is this contour?
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        c_area = cv2.contourArea(contour)
        fill = c_area / rect_area if rect_area > 1 else 0.0

        # Rectangularity: high fill + few vertices
        vert_bonus = 1.0
        if len(approx) == 4:
            vert_bonus = 1.4
        elif len(approx) <= 6:
            vert_bonus = 1.1

        # Area should be substantial but not the full image
        area_quality = area_ratio if area_ratio < 0.95 else area_ratio * 0.3

        total = fill * vert_bonus * area_quality
        if total > best_total_score:
            best_total_score = total
            best_contour = contour
            best_area_ratio = area_ratio
            best_strategy = strategy

    return best_contour, best_area_ratio, best_strategy


# (Legacy alias kept for compatibility)
def find_largest_contour(
    gray: np.ndarray,
    cfg: PreprocConfig,
) -> tuple[np.ndarray | None, float]:
    """Find the largest external contour; return (contour, area_ratio)."""
    c, r = _find_contour_by_otsu(gray)
    return c, r


def approximate_polygon(contour: np.ndarray, epsilon_frac: float = 0.02) -> np.ndarray:
    """Approximate contour as a polygon. Returns Nx1x2 array."""
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon_frac * peri, True)


def rectangle_likeness_score(approx: np.ndarray) -> float:
    """Score 0-1: how well does the polygon approximate a rectangle?

    Best score when exactly 4 vertices and the min-area-rect area closely
    matches the contour area.
    """
    if approx is None or len(approx) < 4:
        return 0.0
    contour_area = cv2.contourArea(approx)
    if contour_area < 1:
        return 0.0
    rect = cv2.minAreaRect(approx)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area < 1:
        return 0.0
    fill_ratio = contour_area / rect_area
    # Bonus for having exactly 4 vertices
    vertex_bonus = 1.0 if len(approx) == 4 else max(0.0, 1.0 - 0.1 * abs(len(approx) - 4))
    return float(np.clip(fill_ratio * vertex_bonus, 0.0, 1.0))


def perspective_skew_score(approx: np.ndarray) -> float:
    """Score 0-1: how skewed / trapezoidal is a quad?

    0 = perfect rectangle, 1 = heavily skewed.
    Works best with exactly 4 vertices.
    """
    if approx is None or len(approx) < 4:
        return 0.0
    pts = approx.reshape(-1, 2).astype(np.float64)
    if len(pts) != 4:
        return 0.0
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = _order_quad(pts)
    # Compare opposite side lengths
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    if top < 1 or bottom < 1 or left < 1 or right < 1:
        return 0.0
    h_ratio = min(top, bottom) / max(top, bottom)
    v_ratio = min(left, right) / max(left, right)
    # Perfect rectangle => both ratios ~1, skew => ratios < 1
    skew = 1.0 - (h_ratio * v_ratio)
    return float(np.clip(skew, 0.0, 1.0))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=pts.dtype)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def ellipse_likeness_score(contour: np.ndarray) -> tuple[float, tuple | None]:
    """Score 0-1: how well does the contour fit an ellipse?

    Returns (score, fitted_ellipse_or_None).
    """
    if contour is None or len(contour) < 5:
        return 0.0, None
    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return 0.0, None
    # Create a mask from the fitted ellipse and compare with contour mask.
    # Guard against degenerate fits where the centre lands far outside the contour —
    # those can produce an enormous (cx*2, cy*2) canvas.  Use the contour's own
    # bounding rect to size the canvas instead, so a bad fit cannot cause a large
    # allocation.
    (cx, cy), (ma, MA), angle = ellipse
    bx, by, bw, bh = cv2.boundingRect(contour)
    max_offset = max(bw, bh)
    if not (bx - max_offset <= cx <= bx + bw + max_offset
            and by - max_offset <= cy <= by + bh + max_offset):
        # Centre is implausibly far from the contour — degenerate fit
        return 0.0, None
    h = int(max(cy * 2, MA) + 20)
    w = int(max(cx * 2, ma) + 20)
    if h < 10 or w < 10 or h > 20000 or w > 20000:
        return 0.0, None
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    intersection = np.count_nonzero(ellipse_mask & contour_mask)
    union = np.count_nonzero(ellipse_mask | contour_mask)
    iou = intersection / union if union > 0 else 0.0
    return float(iou), ellipse


def foreground_isolation_score(gray: np.ndarray, cfg: PreprocConfig) -> float:
    """Score 0-1: how well is the centre region separated from the border?

    Uses the brightness difference between border and centre.
    """
    h, w = gray.shape[:2]
    band = max(1, int(min(h, w) * cfg.border_band_fraction))
    bm = _border_mask(h, w, band)
    cm = _center_mask(h, w, band)
    border_mean = float(np.mean(gray[bm].astype(np.float64))) if np.any(bm) else 128.0
    center_mean = float(np.mean(gray[cm].astype(np.float64))) if np.any(cm) else 128.0
    diff = abs(border_mean - center_mean) / 255.0
    return float(np.clip(diff * 2.0, 0.0, 1.0))


def foreground_isolation_color(
    img: np.ndarray,
    cfg: PreprocConfig,
    *,
    _lab: np.ndarray | None = None,
) -> float:
    """Score 0-1: colour-aware version of foreground isolation.

    Works in Lab space, so coloured borders (pink, cream, etc.) that are
    similar in *brightness* to the photo content still register as different.

    Parameters
    ----------
    _lab : optional pre-computed Lab image (float32).
    """
    if img.ndim != 3:
        return 0.0
    h, w = img.shape[:2]
    band = max(1, int(min(h, w) * cfg.border_band_fraction))
    bm = _border_mask(h, w, band)
    cm = _center_mask(h, w, band)
    lab = _lab if _lab is not None else cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    border_mean = np.mean(lab[bm].astype(np.float32), axis=0)
    center_mean = np.mean(lab[cm].astype(np.float32), axis=0)
    dist = float(np.linalg.norm(border_mean - center_mean))
    # Normalise: 20+ in Lab is clearly different
    return float(np.clip(dist / 30.0, 0.0, 1.0))


# ── Main extraction entry point ──────────────────────────────────────

def extract_features(img: np.ndarray, cfg: PreprocConfig | None = None) -> CropFeatures:
    """Compute all crop-relevant features for *img* (BGR or grayscale).

    Shared intermediates (Lab image, blurred grayscale, border pixels) are
    computed once here and threaded through to every sub-function to avoid
    redundant per-pixel operations.
    """
    if cfg is None:
        cfg = PreprocConfig()
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    feats = CropFeatures()

    # ── Shared intermediates (computed once) ────────────────────────────
    # GaussianBlur is reused by Otsu / edge / Hough strategies
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Lab conversion and border-pixel sample reused by colour-uniformity,
    # border-colour contour strategy, and foreground-isolation-colour
    lab: np.ndarray | None = None
    border_lab_px: np.ndarray | None = None
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        _band = max(4, int(min(h, w) * cfg.border_band_fraction))
        border_lab_px = _sample_border_robust(lab, h, w, _band)

    # ── Feature computation ──────────────────────────────────────────────

    # Border brightness / variance
    feats.border_mean_brightness, feats.border_std_brightness, feats.border_variance = (
        compute_border_stats(gray, cfg)
    )

    # Border colour uniformity (detects coloured mats, not just white)
    feats.border_color_uniformity, feats.border_vs_center_color_dist = (
        compute_border_color_uniformity(img, cfg, _lab=lab, _border_px=border_lab_px)
    )

    # Edge density
    feats.border_edge_density, feats.center_edge_density, feats.edge_density_ratio = (
        compute_edge_densities(gray, cfg)
    )

    # Contour analysis — multi-strategy boundary finder with shared intermediates
    contour, area_ratio, strategy = find_photo_boundary(
        img, gray, cfg,
        _gray_blurred=gray_blurred,
        _lab=lab,
        _border_px=border_lab_px,
    )
    feats.largest_contour = contour
    feats.largest_contour_area_ratio = area_ratio
    feats.contour_strategy = strategy

    if contour is not None:
        approx = approximate_polygon(contour)
        feats.approx_polygon = approx
        feats.polygon_vertex_count = len(approx)
        feats.rectangle_likeness = rectangle_likeness_score(approx)
        feats.perspective_skew_score = perspective_skew_score(approx)

        # Try to find best quad from the contour
        quad = approximate_polygon(contour, epsilon_frac=0.05)
        if len(quad) == 4:
            feats.best_quad = quad

        # Skip ellipse fitting when the contour is already clearly rectangular —
        # the fit would score near 0 anyway and cv2.fitEllipse is non-trivial.
        if feats.rectangle_likeness < 0.85:
            el_score, el_fit = ellipse_likeness_score(contour)
            feats.ellipse_likeness = el_score
            feats.best_ellipse = el_fit

    # Foreground isolation (grayscale + colour)
    feats.foreground_isolation_score = foreground_isolation_score(gray, cfg)
    feats.foreground_isolation_color = foreground_isolation_color(img, cfg, _lab=lab)

    # Aggregate crop likelihood
    feats.crop_likelihood = _compute_crop_likelihood(feats, cfg)

    return feats


def _compute_crop_likelihood(feats: CropFeatures, cfg: PreprocConfig) -> float:
    """Aggregate feature signals into a single 0-1 crop-likelihood score."""
    signals: list[float] = []

    # Bright border is a strong signal
    if feats.border_mean_brightness > cfg.border_brightness_threshold:
        signals.append(0.8)

    # Uniform coloured border with colour distance from centre
    if feats.border_color_uniformity > 0.5 and feats.border_vs_center_color_dist > 0.3:
        signals.append(0.7)

    # Low border variance (uniform border)
    if feats.border_variance < cfg.border_variance_threshold:
        signals.append(0.3)

    # Good contour area ratio with decent rectangularity
    if feats.largest_contour_area_ratio > cfg.min_contour_area_ratio:
        signals.append(0.4 + 0.3 * feats.rectangle_likeness)

    # Foreground isolation (best of grayscale and colour)
    best_isolation = max(feats.foreground_isolation_score, feats.foreground_isolation_color)
    if best_isolation > 0.3:
        signals.append(0.3 * best_isolation)

    # Edge density ratio: more edges in border than center => likely border
    if feats.edge_density_ratio > cfg.edge_density_ratio_threshold:
        signals.append(0.2)

    if not signals:
        return 0.0
    return float(np.clip(np.mean(signals), 0.0, 1.0))
