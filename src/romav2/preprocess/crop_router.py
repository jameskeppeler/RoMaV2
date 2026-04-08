"""Main crop-routing logic: inspect -> classify -> score -> route -> crop.

Public API:
    analyze_and_crop_image(image, enable_crop=True, debug=False, cfg=None)
        -> CropResult
"""

from __future__ import annotations

import cv2
import numpy as np

from .crop_cases import (
    correct_perspective_and_crop,
    crop_rectangular_border,
    crop_white_scan_border,
    detect_irregular_mask_candidate,
)
from .feature_extraction import extract_features
from .types import (
    CaseScore,
    CropCase,
    CropDecision,
    CropFeatures,
    CropResult,
    DebugArtifacts,
    PreprocConfig,
)


# ── Per-case scoring functions ────────────────────────────────────────

def score_white_scan_border(feats: CropFeatures, cfg: PreprocConfig) -> CaseScore:
    """Score the hypothesis: image has a uniform border (white OR coloured).

    Covers white scanner beds, coloured photo mats, pink snapshot borders, etc.
    """
    score = 0.0
    reasons: list[str] = []

    # --- Signal 1: bright white outer band ---
    if feats.border_mean_brightness > cfg.border_brightness_threshold:
        brightness_excess = (feats.border_mean_brightness - cfg.border_brightness_threshold) / (
            255 - cfg.border_brightness_threshold + 1e-6
        )
        score += 0.40 + 0.15 * min(brightness_excess, 1.0)
        reasons.append(f"bright outer band (mean={feats.border_mean_brightness:.0f})")

    # --- Signal 1b: near-pure-white override ---
    # Decorative / scalloped / deckled white borders have irregular edges that
    # drag down uniformity and variance scores.  A very bright outer band
    # (≥ 240, i.e. almost paper-white) is near-certainly a white border
    # regardless of edge texture, so add a small compensating bonus.
    if feats.border_mean_brightness >= 240:
        score += 0.08
        reasons.append(f"near-pure-white band ({feats.border_mean_brightness:.0f} ≥ 240)")

    # --- Signal 2: uniform coloured border (works for pink, cream, etc.) ---
    if feats.border_color_uniformity > 0.5:
        score += 0.25 * feats.border_color_uniformity
        reasons.append(f"uniform border colour (uniformity={feats.border_color_uniformity:.2f})")

    # --- Signal 3: colour distance between border and centre ---
    if feats.border_vs_center_color_dist > 0.3:
        score += 0.20 * feats.border_vs_center_color_dist
        reasons.append(f"border/centre colour differ (dist={feats.border_vs_center_color_dist:.2f})")

    # --- Signal 4: low border variance (uniform texture) ---
    if feats.border_variance < cfg.border_variance_threshold:
        score += 0.10
        reasons.append(f"low-variance border (var={feats.border_variance:.1f})")

    # --- Signal 5: foreground isolation (grey or colour) ---
    best_iso = max(feats.foreground_isolation_score, feats.foreground_isolation_color)
    if best_iso > 0.3:
        score += 0.10 * best_iso
        reasons.append(f"foreground isolation (best={best_iso:.2f})")

    # --- Signal 6: rectangular contour found by boundary finder ---
    if feats.rectangle_likeness > 0.7 and feats.largest_contour_area_ratio > 0.3:
        score += 0.10
        reasons.append(f"rectangular photo boundary found (rect={feats.rectangle_likeness:.2f})")

    return CaseScore(
        case=CropCase.WHITE_SCAN_BORDER,
        score=float(np.clip(score, 0.0, 1.0)),
        rationale="; ".join(reasons) if reasons else "no border signals",
    )


def score_rectangular_border(feats: CropFeatures, cfg: PreprocConfig) -> CaseScore:
    """Score the hypothesis: image has a non-uniform rectangular mat/border.

    This covers cases where the border is not a single colour but the photo
    area is clearly rectangular (e.g. a mat with texture or staining).
    """
    score = 0.0
    reasons: list[str] = []

    # Good rectangle likeness
    if feats.rectangle_likeness > 0.6:
        score += 0.30 * feats.rectangle_likeness
        reasons.append(f"strong rectangle fit ({feats.rectangle_likeness:.2f})")

    # Decent contour area (photo should cover a good fraction but not all)
    if feats.largest_contour_area_ratio > cfg.min_contour_area_ratio:
        if feats.largest_contour_area_ratio < 0.90:
            area_signal = min(feats.largest_contour_area_ratio / 0.5, 1.0)
            score += 0.20 * area_signal
            reasons.append(f"contour covers {feats.largest_contour_area_ratio:.0%} of image")

    # Low skew (unlike phone photos)
    if feats.perspective_skew_score < cfg.perspective_skew_threshold:
        score += 0.10
        reasons.append(f"low skew ({feats.perspective_skew_score:.3f})")

    # Foreground isolation (colour or grey)
    best_iso = max(feats.foreground_isolation_score, feats.foreground_isolation_color)
    if best_iso > 0.25:
        score += 0.15 * best_iso
        reasons.append(f"foreground isolation (best={best_iso:.2f})")

    # Penalise if border is very bright (more likely white_scan case)
    # Use brightness alone as the signal — a scalloped white border may have
    # reduced uniformity but is still unambiguously a scan artifact.
    if feats.border_mean_brightness > cfg.border_brightness_threshold:
        score *= 0.4
        reasons.append(
            f"penalised: bright border ({feats.border_mean_brightness:.0f}) suggests scan"
        )

    # Penalise if no foreground isolation (border and centre look the same)
    if best_iso < 0.10:
        score *= 0.5
        reasons.append(f"penalised: no foreground isolation ({best_iso:.2f})")

    return CaseScore(
        case=CropCase.RECTANGULAR_BORDER,
        score=float(np.clip(score, 0.0, 1.0)),
        rationale="; ".join(reasons) if reasons else "no rectangular-border signals",
    )


def score_phone_photo(feats: CropFeatures, cfg: PreprocConfig) -> CaseScore:
    """Score the hypothesis: phone photo of a physical print (perspective skew)."""
    score = 0.0
    reasons: list[str] = []

    # Strong skew is the primary signal
    if feats.perspective_skew_score > cfg.perspective_skew_threshold:
        skew_signal = min(feats.perspective_skew_score / 0.4, 1.0)
        score += 0.45 * skew_signal
        reasons.append(f"quadrilateral strongly skewed ({feats.perspective_skew_score:.3f})")

    # 4-vertex polygon (the photo edges)
    if feats.polygon_vertex_count == 4:
        score += 0.20
        reasons.append("dominant 4-corner contour found")

    # Good contour area
    if feats.largest_contour_area_ratio > cfg.min_contour_area_ratio:
        score += 0.15
        reasons.append(f"contour area {feats.largest_contour_area_ratio:.0%}")

    # Foreground isolation
    best_iso = max(feats.foreground_isolation_score, feats.foreground_isolation_color)
    if best_iso > 0.3:
        score += 0.10 * best_iso
        reasons.append(f"foreground isolation ({best_iso:.2f})")

    # Must have a quad
    if feats.best_quad is None:
        score *= 0.3
        reasons.append("penalised: no quad detected")

    # Penalise when the outer band is clearly a uniform bright border.
    # Scanned photos with decorative/deckled edges can produce a slightly
    # tilted quad, but they are NOT phone photos of a print.
    if feats.border_mean_brightness > cfg.border_brightness_threshold:
        score *= 0.35
        reasons.append(
            f"penalised: bright uniform border ({feats.border_mean_brightness:.0f}) "
            "indicates scan, not phone photo"
        )
    elif feats.border_color_uniformity > 0.60:
        # Uniform coloured mat (card mount, decorative border) — not a phone photo.
        # Require only high uniformity; don't also gate on color distance because
        # the distance metric can miss certain hue combinations.
        score *= 0.40
        reasons.append(
            f"penalised: uniform coloured border (uniformity={feats.border_color_uniformity:.2f}) "
            "indicates scan/mat, not phone photo"
        )

    return CaseScore(
        case=CropCase.PHONE_PHOTO_PERSPECTIVE,
        score=float(np.clip(score, 0.0, 1.0)),
        rationale="; ".join(reasons) if reasons else "no phone-photo signals",
    )


def score_irregular_candidate(feats: CropFeatures, cfg: PreprocConfig) -> CaseScore:
    """Score the hypothesis: oval vignette / circular opening / cabinet card."""
    score = 0.0
    reasons: list[str] = []

    # Ellipse likeness is the primary signal
    if feats.ellipse_likeness > cfg.ellipse_fit_threshold:
        score += 0.50 * feats.ellipse_likeness
        reasons.append(f"strong ellipse fit ({feats.ellipse_likeness:.2f})")
    elif feats.ellipse_likeness > 0.4:
        score += 0.20 * feats.ellipse_likeness
        reasons.append(f"moderate ellipse fit ({feats.ellipse_likeness:.2f})")

    # Poor rectangle fit reinforces (it's NOT a rectangle)
    if feats.rectangle_likeness < 0.5:
        score += 0.15
        reasons.append(f"poor rectangle fit ({feats.rectangle_likeness:.2f}) supports oval hypothesis")
    elif feats.rectangle_likeness > 0.75:
        # High rectangle likeness STRONGLY argues against oval
        score *= 0.3
        reasons.append(f"penalised: high rectangle fit ({feats.rectangle_likeness:.2f}) contradicts oval")

    # Foreground isolation
    best_iso = max(feats.foreground_isolation_score, feats.foreground_isolation_color)
    if best_iso > 0.3:
        score += 0.10 * best_iso
        reasons.append(f"foreground isolation ({best_iso:.2f})")

    # Contour area — penalise if contour fills nearly the whole image
    if feats.largest_contour_area_ratio > cfg.min_contour_area_ratio:
        if feats.largest_contour_area_ratio < 0.80:
            score += 0.10
            reasons.append(f"enclosed central region ({feats.largest_contour_area_ratio:.0%})")
        else:
            score *= 0.5
            reasons.append(
                f"contour fills {feats.largest_contour_area_ratio:.0%} of image "
                "(near full-frame, penalised)"
            )

    # Penalise when border and centre look the same (no real surround)
    if best_iso < 0.10:
        score *= 0.5
        reasons.append(f"no foreground isolation ({best_iso:.2f}), penalised")

    return CaseScore(
        case=CropCase.IRREGULAR_MASK_CANDIDATE,
        score=float(np.clip(score, 0.0, 1.0)),
        rationale="; ".join(reasons) if reasons else "no irregular-shape signals",
    )


def score_no_crop(feats: CropFeatures, cfg: PreprocConfig) -> CaseScore:
    """Score the hypothesis: image needs no cropping."""
    score = 0.0
    reasons: list[str] = []

    # Low crop likelihood => no crop
    if feats.crop_likelihood < 0.3:
        score += 0.50
        reasons.append(f"low crop likelihood ({feats.crop_likelihood:.2f})")

    # Contour covers most of the image (photo IS the image)
    if feats.largest_contour_area_ratio > 0.90:
        score += 0.25
        reasons.append(f"contour fills {feats.largest_contour_area_ratio:.0%} of image (full-frame)")

    # Low border colour uniformity (border is not a uniform colour)
    if feats.border_color_uniformity < 0.3:
        score += 0.10
        reasons.append(f"non-uniform border (uniformity={feats.border_color_uniformity:.2f})")

    # Low edge-density ratio
    if feats.edge_density_ratio < 1.2:
        score += 0.10
        reasons.append(f"uniform edge distribution (ratio={feats.edge_density_ratio:.2f})")

    return CaseScore(
        case=CropCase.NO_CROP,
        score=float(np.clip(score, 0.0, 1.0)),
        rationale="; ".join(reasons) if reasons else "some crop signals present",
    )


# ── Iterative re-crop ─────────────────────────────────────────────────

def _iterative_recrop(
    img: np.ndarray,
    cfg: PreprocConfig,
    notes: list[str],
    max_passes: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Re-analyse a cropped image and crop again if another border remains.

    Handles multi-layer borders (e.g. pink outer + white inner) by
    running the scoring pipeline on the already-cropped result.  Only
    re-crops if the new analysis is confident enough.
    """
    current = img
    for pass_idx in range(max_passes):
        # Respect the working-resolution setting (same downscale → upscale path
        # as the top-level analyze_and_crop_image call).
        _h2, _w2 = current.shape[:2]
        _min2 = min(_h2, _w2)
        _working2 = current
        _inv2 = 1.0
        if cfg.max_working_res and _min2 > cfg.max_working_res:
            _s2 = cfg.max_working_res / _min2
            _working2 = cv2.resize(
                current,
                (max(1, round(_w2 * _s2)), max(1, round(_h2 * _s2))),
                interpolation=cv2.INTER_AREA,
            )
            _inv2 = _min2 / cfg.max_working_res
        feats2 = extract_features(_working2, cfg)
        if _inv2 != 1.0:
            feats2 = _scale_features(feats2, _inv2)
        scores2 = [
            score_white_scan_border(feats2, cfg),
            score_rectangular_border(feats2, cfg),
            score_no_crop(feats2, cfg),
        ]
        best2 = max(scores2, key=lambda s: s.score)

        # Only re-crop if confident there's still a clear border.
        if best2.case not in (CropCase.WHITE_SCAN_BORDER, CropCase.RECTANGULAR_BORDER):
            break
        # Threshold is 0.50 (not 0.60) because:
        # (a) the main crop already confirmed a border was present, so we can
        #     be slightly more permissive on subsequent passes; and
        # (b) multi-layer borders leave a thin mixed fringe after the first
        #     crop that dilutes brightness/uniformity signals below 0.60.
        if best2.score < 0.50:
            break

        # Require strong foreground isolation (border clearly differs from centre)
        best_iso = max(feats2.foreground_isolation_score, feats2.foreground_isolation_color)
        if best_iso < 0.20:
            break

        # For WHITE_SCAN_BORDER also require that the border is truly uniform —
        # UNLESS the border is very bright (near paper-white).  Scalloped /
        # deckled edges on white borders let dark photo pixels bleed into the
        # border sample band, dragging uniformity below 0.5 even though the
        # border is unambiguously white paper.  Trust brightness over uniformity
        # when the outer band is almost pure white (mean ≥ 230).
        if best2.case == CropCase.WHITE_SCAN_BORDER:
            is_clearly_white = feats2.border_mean_brightness >= 230
            if not is_clearly_white and feats2.border_color_uniformity < 0.5:
                break

        if best2.case == CropCase.WHITE_SCAN_BORDER:
            recropped, pass_notes = crop_white_scan_border(current, feats2, cfg)
        else:
            recropped, pass_notes = crop_rectangular_border(current, feats2, cfg)

        if recropped is current:
            break

        # Verify the re-crop removes a plausible fraction of area.
        # Lower bound (3%): prevents trivial no-op re-crops.
        # Upper bound (65%): allows thick multi-layer borders (e.g. wide pink
        # outer + wide white inner) where the second pass may need to remove
        # close to half the already-cropped image area.  The score / isolation
        # guards above are the primary safety net against over-cropping.
        rh, rw = recropped.shape[:2]
        oh, ow = current.shape[:2]
        removed_frac = 1.0 - (rh * rw) / (oh * ow)
        if removed_frac < 0.03 or removed_frac > 0.65:
            break

        notes.append(f"Pass {pass_idx + 2}: re-cropped remaining border "
                     f"(removed {removed_frac:.1%}, iso={best_iso:.2f})")
        notes.extend(pass_notes)
        current = recropped

    return current, notes


# ── Coordinate scaling ───────────────────────────────────────────────

def _scale_features(feats: CropFeatures, scale: float) -> CropFeatures:
    """Scale pixel-space feature coordinates by *scale*.

    Used to map contour / quad / ellipse coordinates from the working
    (downscaled) resolution back to the original image resolution.
    Scalar ratios and scores are resolution-independent and unchanged.
    """
    if scale == 1.0:
        return feats
    from copy import copy
    scaled = copy(feats)
    s32 = np.float32(scale)
    if feats.largest_contour is not None:
        scaled.largest_contour = np.round(feats.largest_contour * s32).astype(np.int32)
    if feats.approx_polygon is not None:
        scaled.approx_polygon = np.round(feats.approx_polygon * s32).astype(np.int32)
    if feats.best_quad is not None:
        scaled.best_quad = np.round(feats.best_quad * s32).astype(np.int32)
    if feats.best_ellipse is not None:
        (cx, cy), (MA, ma), angle = feats.best_ellipse
        scaled.best_ellipse = (
            (cx * scale, cy * scale),
            (MA * scale, ma * scale),
            angle,
        )
    return scaled


# ── Router ────────────────────────────────────────────────────────────

def _choose_best_case(scores: list[CaseScore]) -> CaseScore:
    """Return the highest-scoring case."""
    return max(scores, key=lambda s: s.score)


def analyze_and_crop_image(
    image: np.ndarray,
    *,
    enable_crop: bool = True,
    debug: bool = False,
    cfg: PreprocConfig | None = None,
) -> CropResult:
    """Full pipeline: extract features -> score cases -> apply crop.

    Parameters
    ----------
    image : BGR numpy array
    enable_crop : if False, analyse only (no modification)
    debug : if True, populate DebugArtifacts
    cfg : optional config overrides

    Returns
    -------
    CropResult with the (possibly cropped) image, decision, and debug info.
    """
    if cfg is None:
        cfg = PreprocConfig()

    original = image.copy()
    debug_arts = DebugArtifacts(original=original) if debug else None

    # Step 1: optionally downsample to a working resolution for feature extraction.
    # Feature detection does not need full resolution — border patterns, contours,
    # and texture signals are equally visible on a smaller image — but per-pixel
    # operations scale quadratically, so halving the shorter dimension gives ~4×
    # speedup with zero loss in crop accuracy.  The extracted coordinates are
    # scaled back to the original resolution before the crop handlers run.
    h, w = image.shape[:2]
    min_dim = min(h, w)
    inv_scale = 1.0          # multiply working-res coords by this → original coords
    working = image
    if cfg.max_working_res and min_dim > cfg.max_working_res:
        scale = cfg.max_working_res / min_dim
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        working = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        inv_scale = min_dim / cfg.max_working_res

    feats = extract_features(working, cfg)

    # Scale pixel-space coordinates back to original image dimensions
    if inv_scale != 1.0:
        feats = _scale_features(feats, inv_scale)

    # Step 2: score every case
    all_scores = [
        score_white_scan_border(feats, cfg),
        score_rectangular_border(feats, cfg),
        score_phone_photo(feats, cfg),
        score_irregular_candidate(feats, cfg),
        score_no_crop(feats, cfg),
    ]

    best = _choose_best_case(all_scores)

    # Step 3: decide whether to act
    if not enable_crop or best.case == CropCase.NO_CROP or best.score < cfg.min_crop_confidence:
        label = best.case if best.score >= cfg.min_crop_confidence else CropCase.UNCERTAIN
        decision = CropDecision(
            case_label=label,
            confidence=best.score,
            selected_method="none",
            rationale=best.rationale if label != CropCase.UNCERTAIN else (
                f"Best case was {best.case.value} at {best.score:.2f} "
                f"but below threshold {cfg.min_crop_confidence}"
            ),
            all_scores=all_scores,
        )
        return CropResult(
            image=image,
            original=original,
            decision=decision,
            features=feats,
            debug=debug_arts,
            was_cropped=False,
        )

    # Step 4: apply the winning case handler
    result_img = image
    notes: list[str] = []
    method = best.case.value

    if best.case == CropCase.WHITE_SCAN_BORDER:
        result_img, notes = crop_white_scan_border(image, feats, cfg)
    elif best.case == CropCase.RECTANGULAR_BORDER:
        result_img, notes = crop_rectangular_border(image, feats, cfg)
    elif best.case == CropCase.PHONE_PHOTO_PERSPECTIVE:
        result_img, notes = correct_perspective_and_crop(image, feats, cfg)
    elif best.case == CropCase.IRREGULAR_MASK_CANDIDATE:
        result_img, mask, notes = detect_irregular_mask_candidate(image, feats, cfg)
        if debug_arts is not None:
            debug_arts.border_mask = mask

    was_cropped = result_img is not image

    # Step 5: iterative re-crop for multi-layer borders
    # After removing one border layer, check if there's another uniform border
    # remaining (e.g. pink outer removed -> white inner still present).
    # Only do this for border-removal cases, not perspective/irregular.
    if was_cropped and best.case in (CropCase.WHITE_SCAN_BORDER, CropCase.RECTANGULAR_BORDER):
        result_img, notes = _iterative_recrop(result_img, cfg, notes, max_passes=2)
        # was_cropped stays True — first crop already happened

    if debug_arts is not None:
        debug_arts.crop_preview = result_img
        debug_arts.notes = notes

    decision = CropDecision(
        case_label=best.case,
        confidence=best.score,
        selected_method=method,
        rationale=best.rationale + (" | " + "; ".join(notes) if notes else ""),
        all_scores=all_scores,
    )

    return CropResult(
        image=result_img,
        original=original,
        decision=decision,
        features=feats,
        debug=debug_arts,
        was_cropped=was_cropped,
    )
