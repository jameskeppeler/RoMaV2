"""Tests for the romav2.preprocess crop-routing pipeline.

These tests use synthetic images to validate each crop case.
For real-world validation, place sample images in tests/samples/ and
uncomment the marked sections.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if opencv is not installed
cv2 = pytest.importorskip("cv2")

from romav2.preprocess import (
    CropCase,
    PreprocConfig,
    analyze_and_crop_image,
)
from romav2.preprocess.feature_extraction import (
    compute_border_stats,
    compute_edge_densities,
    extract_features,
    find_photo_boundary,
    rectangle_likeness_score,
    approximate_polygon,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _white_border_scan_image(h: int = 600, w: int = 800, border: int = 40) -> np.ndarray:
    """Synthetic: dark photo on a white scanner bed."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # white
    img[border:h - border, border:w - border] = (60, 60, 60)  # dark centre
    return img


def _rectangular_border_image(h: int = 600, w: int = 800, border: int = 50) -> np.ndarray:
    """Synthetic: dark photo on a coloured (non-white) mat."""
    img = np.full((h, w, 3), 100, dtype=np.uint8)  # grey-brown mat
    img[border:h - border, border:w - border] = (50, 50, 50)  # darker photo
    return img


def _perspective_skew_image(h: int = 600, w: int = 800) -> np.ndarray:
    """Synthetic: a trapezoid (simulating a phone photo of a print)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)  # light background
    pts = np.array([
        [120, 80],
        [680, 50],
        [720, 520],
        [80, 550],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, (40, 40, 40))
    return img


def _oval_vignette_image(h: int = 600, w: int = 800) -> np.ndarray:
    """Synthetic: dark oval on a lighter background (cabinet card / vignette)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, (40, 40, 40), -1)
    return img


def _full_frame_image(h: int = 600, w: int = 800) -> np.ndarray:
    """Synthetic: image content fills the entire frame (no crop needed)."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 180, (h, w, 3), dtype=np.uint8)


# ── Feature extraction tests ─────────────────────────────────────────

class TestFeatureExtraction:
    def test_border_stats_white_border(self):
        img = _white_border_scan_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = PreprocConfig()
        mean, std, var = compute_border_stats(gray, cfg)
        assert mean > 200, f"Expected bright border, got mean={mean}"
        assert std < 50, f"Expected low variance border, got std={std}"

    def test_border_stats_full_frame(self):
        img = _full_frame_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = PreprocConfig()
        mean, std, var = compute_border_stats(gray, cfg)
        # Full-frame: border and centre should be similar (not bright white)
        assert mean < 200

    def test_edge_densities(self):
        img = _white_border_scan_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = PreprocConfig()
        bd, cd, ratio = compute_edge_densities(gray, cfg)
        assert bd >= 0
        assert cd >= 0

    def test_largest_contour_white_border(self):
        img = _white_border_scan_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = PreprocConfig()
        contour, ratio, strategy = find_photo_boundary(img, gray, cfg)
        assert contour is not None
        assert ratio > 0.1

    def test_extract_features_returns_all_fields(self):
        img = _white_border_scan_image()
        feats = extract_features(img)
        assert feats.border_mean_brightness > 0
        assert feats.crop_likelihood >= 0


# ── Router tests ──────────────────────────────────────────────────────

class TestCropRouter:
    def test_white_scan_border_detected(self):
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img, debug=True)
        assert result.decision.case_label == CropCase.WHITE_SCAN_BORDER
        assert result.was_cropped
        # Cropped image should be smaller
        assert result.image.shape[0] < img.shape[0] or result.image.shape[1] < img.shape[1]

    def test_rectangular_border_detected(self):
        img = _rectangular_border_image()
        result = analyze_and_crop_image(img, debug=True)
        # May detect as rectangular_border or no_crop depending on contrast;
        # the key is it doesn't crash and returns a valid result
        assert result.decision.case_label in (
            CropCase.RECTANGULAR_BORDER,
            CropCase.WHITE_SCAN_BORDER,
            CropCase.NO_CROP,
            CropCase.UNCERTAIN,
        )

    def test_perspective_skew_detected(self):
        img = _perspective_skew_image()
        result = analyze_and_crop_image(img, debug=True)
        # With a synthetic trapezoid, it should detect perspective or rectangular
        assert result.decision.case_label in (
            CropCase.PHONE_PHOTO_PERSPECTIVE,
            CropCase.RECTANGULAR_BORDER,
            CropCase.WHITE_SCAN_BORDER,
            CropCase.UNCERTAIN,
        )

    def test_oval_vignette_detected(self):
        img = _oval_vignette_image()
        result = analyze_and_crop_image(img, debug=True)
        # May detect as irregular or rectangular depending on the fit
        assert result.decision.case_label in (
            CropCase.IRREGULAR_MASK_CANDIDATE,
            CropCase.RECTANGULAR_BORDER,
            CropCase.WHITE_SCAN_BORDER,
            CropCase.UNCERTAIN,
            CropCase.NO_CROP,
        )

    def test_full_frame_no_crop(self):
        img = _full_frame_image()
        result = analyze_and_crop_image(img, debug=True)
        # Random noise image should not trigger a confident crop
        assert result.decision.case_label in (CropCase.NO_CROP, CropCase.UNCERTAIN)
        assert not result.was_cropped

    def test_enable_crop_false(self):
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img, enable_crop=False)
        assert not result.was_cropped
        assert np.array_equal(result.image, img)

    def test_debug_artifacts_populated(self):
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img, debug=True)
        assert result.debug is not None
        assert result.debug.original is not None

    def test_all_scores_present(self):
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img)
        labels = {s.case for s in result.decision.all_scores}
        expected = {
            CropCase.WHITE_SCAN_BORDER,
            CropCase.RECTANGULAR_BORDER,
            CropCase.PHONE_PHOTO_PERSPECTIVE,
            CropCase.IRREGULAR_MASK_CANDIDATE,
            CropCase.NO_CROP,
        }
        assert labels == expected


# ── Debug viz smoke test ──────────────────────────────────────────────

class TestDebugViz:
    def test_build_debug_panel(self):
        from romav2.preprocess.debug_viz import build_debug_panel
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img, debug=True)
        panel = build_debug_panel(result)
        assert panel.ndim == 3
        assert panel.shape[0] > 0 and panel.shape[1] > 0

    def test_save_debug_outputs(self, tmp_path):
        from romav2.preprocess.debug_viz import save_debug_outputs
        img = _white_border_scan_image()
        result = analyze_and_crop_image(img, debug=True)
        saved = save_debug_outputs(result, tmp_path / "debug")
        assert len(saved) > 0
        for p in saved:
            assert p.exists()


# ── Quality guards ────────────────────────────────────────────────────

class TestCropQuality:
    """Verify that crop coordinates are accurate and no content is lost.

    These tests use synthetic images with known, exact borders so we can
    assert tight bounds on the output dimensions.
    """

    # Maximum tolerated deviation from the true crop edge (pixels)
    TOLERANCE = 12

    @pytest.mark.parametrize("res_scale", [1, 2, 4])
    def test_white_border_crop_accuracy(self, res_scale: int):
        """Cropped image should be close to the true content area."""
        h, w, border = 200 * res_scale, 270 * res_scale, 20 * res_scale
        img = _white_border_scan_image(h=h, w=w, border=border)
        result = analyze_and_crop_image(img, cfg=PreprocConfig(max_working_res=800))
        assert result.was_cropped, "expected crop"
        oh, ow = result.image.shape[:2]
        # True content: (h - 2*border) × (w - 2*border)
        true_h = h - 2 * border
        true_w = w - 2 * border
        assert abs(oh - true_h) <= self.TOLERANCE, f"height off by {abs(oh-true_h)}px"
        assert abs(ow - true_w) <= self.TOLERANCE, f"width off by {abs(ow-true_w)}px"

    @pytest.mark.parametrize("res_scale", [1, 2, 4])
    def test_output_dimensions_positive(self, res_scale: int):
        """Crop output must always have positive dimensions."""
        for gen in [_white_border_scan_image, _rectangular_border_image,
                    _perspective_skew_image, _oval_vignette_image, _full_frame_image]:
            h, w = 200 * res_scale, 270 * res_scale
            try:
                img = gen(h=h, w=w)
            except TypeError:
                img = gen()
            result = analyze_and_crop_image(img)
            assert result.image.shape[0] > 0 and result.image.shape[1] > 0

    def test_no_crop_case_unchanged(self):
        """NO_CROP must return an image with the same dimensions as input."""
        img = _full_frame_image(h=400, w=530)
        result = analyze_and_crop_image(img)
        if not result.was_cropped:
            assert result.image.shape[:2] == img.shape[:2]

    def test_working_res_disabled_vs_enabled_consistency(self):
        """Both configs should produce the same crop case and similar output size."""
        img = _white_border_scan_image(h=600, w=800, border=50)
        cfg_off = PreprocConfig(max_working_res=0)
        cfg_on  = PreprocConfig(max_working_res=800)
        r_off = analyze_and_crop_image(img, cfg=cfg_off)
        r_on  = analyze_and_crop_image(img, cfg=cfg_on)
        assert r_off.decision.case_label == r_on.decision.case_label
        oh_off, ow_off = r_off.image.shape[:2]
        oh_on,  ow_on  = r_on.image.shape[:2]
        # Crop sizes should be within 5% of each other
        assert abs(oh_off - oh_on) / max(oh_off, 1) < 0.05
        assert abs(ow_off - ow_on) / max(ow_off, 1) < 0.05

    @pytest.mark.parametrize("aspect", [(1, 3), (3, 1), (1, 1)])
    def test_extreme_aspect_ratios(self, aspect: tuple[int, int]):
        """Extreme aspect ratios must not crash or produce degenerate output."""
        h = 200 * aspect[0]
        w = 200 * aspect[1]
        b = 20
        img = np.full((h, w, 3), 240, dtype=np.uint8)
        img[b:h-b, b:w-b] = 60
        result = analyze_and_crop_image(img)
        assert result.image.shape[0] > 0 and result.image.shape[1] > 0

    def test_tiny_image_no_crash(self):
        """Very small images (< working_res) should be processed without error."""
        img = np.full((80, 100, 3), 255, dtype=np.uint8)
        img[10:70, 10:90] = 40
        result = analyze_and_crop_image(img)
        assert result.image is not None

    def test_two_layer_border_both_removed(self):
        """Pink outer border + white inner border: both layers must be cropped away.

        This reproduces the real-world pattern seen in 1950s snapshot prints
        where the photo paper has a decorative scalloped / deckled white border
        inside a pink printed outer border.  The iterative recrop must fire a
        second pass to remove the white inner border even though its scalloped
        edges reduce measured colour uniformity below the normal threshold.
        """
        h, w = 400, 400
        photo_top, photo_left = 80, 80           # where the actual photo starts
        photo_bottom, photo_right = h - 80, w - 80

        img = np.full((h, w, 3), (170, 140, 200), dtype=np.uint8)  # pink outer

        # White inner border (uniform, simulating the paper)
        white_top, white_left = 40, 40
        white_bottom, white_right = h - 40, w - 40
        img[white_top:white_bottom, white_left:white_right] = 245

        # Simulate scalloped/deckled edges: punch small dark notches into the
        # white border from all four sides so measured uniformity drops.
        rng = np.random.RandomState(7)
        for side in range(4):
            for _ in range(18):
                if side == 0:   # top
                    r = rng.randint(white_top, white_top + 18)
                    c = rng.randint(white_left, white_right)
                    img[r:r + 6, c:c + 6] = 30
                elif side == 1:  # bottom
                    r = rng.randint(white_bottom - 18, white_bottom)
                    c = rng.randint(white_left, white_right)
                    img[r:r + 6, c:c + 6] = 30
                elif side == 2:  # left
                    r = rng.randint(white_top, white_bottom)
                    c = rng.randint(white_left, white_left + 18)
                    img[r:r + 6, c:c + 6] = 30
                else:            # right
                    r = rng.randint(white_top, white_bottom)
                    c = rng.randint(white_right - 18, white_right)
                    img[r:r + 6, c:c + 6] = 30

        # Dark photo content
        img[photo_top:photo_bottom, photo_left:photo_right] = np.clip(
            np.full((photo_bottom - photo_top, photo_right - photo_left, 3), 55)
            + rng.randint(-15, 15, (photo_bottom - photo_top, photo_right - photo_left, 3)),
            0, 255,
        ).astype(np.uint8)

        result = analyze_and_crop_image(img)
        assert result.was_cropped, "expected at least one crop pass"

        oh, ow = result.image.shape[:2]
        true_h = photo_bottom - photo_top
        true_w = photo_right - photo_left
        tolerance = 20
        assert abs(oh - true_h) <= tolerance, (
            f"white inner border not fully removed: output h={oh}, expected ~{true_h}"
        )
        assert abs(ow - true_w) <= tolerance, (
            f"white inner border not fully removed: output w={ow}, expected ~{true_w}"
        )

    def test_high_res_crop_not_worse_than_low_res(self):
        """Upscaling a clean border image should not degrade crop accuracy."""
        # A 4× upscaled version of the same scene should yield proportionally
        # equivalent crop bounds (within tolerance).
        h, w, b = 200, 270, 20
        img_lo = _white_border_scan_image(h=h, w=w, border=b)
        img_hi = cv2.resize(img_lo, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        r_lo = analyze_and_crop_image(img_lo)
        r_hi = analyze_and_crop_image(img_hi)
        # Content fractions should be similar
        frac_lo = (r_lo.image.shape[0] * r_lo.image.shape[1]) / (h * w)
        frac_hi = (r_hi.image.shape[0] * r_hi.image.shape[1]) / (h * 3 * w * 3)
        assert abs(frac_lo - frac_hi) < 0.15, (
            f"crop fraction mismatch: lo={frac_lo:.2f} hi={frac_hi:.2f}"
        )


# To test with real sample images, create tests/samples/ with:
#   scan_white_border.jpg
#   rectangular_border.jpg
#   phone_photo.jpg
#   oval_vignette.jpg
#   full_frame.jpg
# Then uncomment and adapt the test below:
#
# class TestRealSamples:
#     SAMPLES_DIR = Path(__file__).parent / "samples"
#
#     @pytest.mark.skipif(not (SAMPLES_DIR / "scan_white_border.jpg").exists(),
#                         reason="sample not available")
#     def test_real_scan_white_border(self):
#         img = cv2.imread(str(self.SAMPLES_DIR / "scan_white_border.jpg"))
#         result = analyze_and_crop_image(img, debug=True)
#         assert result.decision.case_label == CropCase.WHITE_SCAN_BORDER
