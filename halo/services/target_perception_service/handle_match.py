from __future__ import annotations

import math
import re

from halo.services.target_perception_service.vlm_parser import VlmDetection


def find_detection_by_handle(
    target_handle: str,
    detections: list[VlmDetection],
    *,
    reference_center_px: tuple[float, float] | None = None,
) -> VlmDetection | None:
    """Find a detection for *target_handle* using service fuzzy-match semantics.

    1. Exact handle match.
    2. Fuzzy fallback: strip trailing ``_NN`` and match on prefix.
       If multiple candidates share the prefix, pick the first.
    """
    exact = [d for d in detections if d.handle == target_handle]
    if exact:
        return _choose_candidate(exact, reference_center_px)

    prefix = re.sub(r"_\d+$", "", target_handle)
    candidates = [d for d in detections if re.sub(r"_\d+$", "", d.handle) == prefix]
    return _choose_candidate(candidates, reference_center_px)


def _choose_candidate(
    detections: list[VlmDetection],
    reference_center_px: tuple[float, float] | None,
) -> VlmDetection | None:
    if not detections:
        return None
    if reference_center_px is None:
        return detections[0]

    cx, cy = reference_center_px
    return min(
        detections,
        key=lambda d: math.hypot(d.centroid[0] - cx, d.centroid[1] - cy),
    )
