"""Typed data structures for the photo preprocessing / crop-routing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CropCase(str, Enum):
    """Labels for the crop-routing decision."""
    NO_CROP = "no_crop"
    WHITE_SCAN_BORDER = "white_scan_border"
    RECTANGULAR_BORDER = "rectangular_border"
    PHONE_PHOTO_PERSPECTIVE = "phone_photo_perspective"
    IRREGULAR_MASK_CANDIDATE = "irregular_mask_candidate"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Configurable thresholds  (single source of truth for tuning)
# ---------------------------------------------------------------------------

@dataclass
class PreprocConfig:
    """All tuneable thresholds for the crop-decision pipeline."""

    # Minimum overall confidence to actually apply any crop/transform.
    min_crop_confidence: float = 0.55

    # Border brightness: mean of outer-band pixels (0-255 scale).
    border_brightness_threshold: float = 220.0

    # Border variance: low variance => uniform border (white scan bed).
    border_variance_threshold: float = 180.0

    # Minimum fraction of image area that the largest contour must occupy.
    min_contour_area_ratio: float = 0.10

    # Perspective skew: how far from a perfect rectangle (0 = perfect).
    perspective_skew_threshold: float = 0.15

    # Outer-band width as fraction of the shorter image dimension.
    border_band_fraction: float = 0.05

    # Edge-density ratio (border/center) threshold for border detection.
    edge_density_ratio_threshold: float = 2.0

    # Ellipse-likeness: how well the largest contour fits an ellipse (0-1).
    ellipse_fit_threshold: float = 0.70

    # Working resolution: downsample the shorter image dimension to this value
    # before feature extraction, then scale crop coordinates back to original size.
    # Dramatically reduces cost for high-resolution inputs (scanners, phones).
    # Set to 0 to disable downscaling entirely.
    max_working_res: int = 800


# ---------------------------------------------------------------------------
# Feature / decision / result containers
# ---------------------------------------------------------------------------

@dataclass
class CropFeatures:
    """Image features relevant to crop-routing decisions."""

    # Outer-band statistics
    border_mean_brightness: float = 0.0
    border_std_brightness: float = 0.0
    border_variance: float = 0.0

    # Border colour uniformity (detects coloured mats, not just white)
    border_color_uniformity: float = 0.0
    border_vs_center_color_dist: float = 0.0

    # Edge density
    border_edge_density: float = 0.0
    center_edge_density: float = 0.0
    edge_density_ratio: float = 0.0

    # Contour analysis
    largest_contour_area_ratio: float = 0.0
    polygon_vertex_count: int = 0
    rectangle_likeness: float = 0.0
    perspective_skew_score: float = 0.0
    ellipse_likeness: float = 0.0
    contour_strategy: str = ""

    # Foreground isolation
    foreground_isolation_score: float = 0.0
    foreground_isolation_color: float = 0.0

    # Overall
    crop_likelihood: float = 0.0

    # Raw intermediates (not serialised to JSON)
    largest_contour: Any = field(default=None, repr=False)
    approx_polygon: Any = field(default=None, repr=False)
    best_quad: Any = field(default=None, repr=False)
    best_ellipse: Any = field(default=None, repr=False)


@dataclass
class CaseScore:
    """Score + rationale for a single crop-case hypothesis."""
    case: CropCase
    score: float
    rationale: str


@dataclass
class CropDecision:
    """The routing decision: which case won and why."""
    case_label: CropCase
    confidence: float
    selected_method: str
    rationale: str
    all_scores: list[CaseScore] = field(default_factory=list)


@dataclass
class DebugArtifacts:
    """Optional debug images / metadata produced during analysis."""
    original: np.ndarray | None = None
    border_mask: np.ndarray | None = None
    contour_overlay: np.ndarray | None = None
    quad_overlay: np.ndarray | None = None
    ellipse_overlay: np.ndarray | None = None
    crop_preview: np.ndarray | None = None
    side_by_side: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CropResult:
    """Full output of the analyze-and-crop pipeline."""
    image: np.ndarray
    original: np.ndarray
    decision: CropDecision
    features: CropFeatures
    debug: DebugArtifacts | None = None
    was_cropped: bool = False
