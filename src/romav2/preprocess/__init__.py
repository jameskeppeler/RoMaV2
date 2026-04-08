"""Photo preprocessing / crop-routing pipeline.

Public API::

    from romav2.preprocess import analyze_and_crop_image, PreprocConfig

    result = analyze_and_crop_image(bgr_image, debug=True)
    print(result.decision.case_label, result.decision.confidence)
"""

from .crop_router import analyze_and_crop_image
from .types import (
    CaseScore,
    CropCase,
    CropDecision,
    CropFeatures,
    CropResult,
    DebugArtifacts,
    PreprocConfig,
)

__all__ = [
    "analyze_and_crop_image",
    "CaseScore",
    "CropCase",
    "CropDecision",
    "CropFeatures",
    "CropResult",
    "DebugArtifacts",
    "PreprocConfig",
]
