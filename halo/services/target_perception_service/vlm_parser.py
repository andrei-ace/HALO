from __future__ import annotations

from dataclasses import dataclass

# VLM models (e.g. qwen3-vl:30B) emit coordinates as integers in [0, 1000].
# This module normalises them to [0.0, 1.0] before any downstream use.

_VLM_COORD_MAX = 1000.0


@dataclass(frozen=True)
class VlmDetection:
    handle: str
    label: str
    # Normalised [0, 1] bounding box: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]
    # Normalised [0, 1] image centroid: (cx, cy)
    centroid: tuple[float, float]
    estimated_depth_m: float
    confidence: float
    is_graspable: bool


def parse_vlm_response(response: dict) -> list[VlmDetection]:
    """
    Parse a raw VLM JSON response into a list of VlmDetection objects.

    Coordinates in the response are expected in [0, 1000] and are normalised
    to [0.0, 1.0] here.  All other fields are passed through unchanged.
    """
    detections: list[VlmDetection] = []
    for det in response.get("detections", []):
        x1, y1, x2, y2 = det["bounding_box"]
        cx, cy = det["centroid_xy"]
        detections.append(
            VlmDetection(
                handle=det["handle"],
                label=det["label"],
                bbox=(
                    x1 / _VLM_COORD_MAX,
                    y1 / _VLM_COORD_MAX,
                    x2 / _VLM_COORD_MAX,
                    y2 / _VLM_COORD_MAX,
                ),
                centroid=(cx / _VLM_COORD_MAX, cy / _VLM_COORD_MAX),
                estimated_depth_m=det["estimated_depth_m"],
                confidence=det["confidence"],
                is_graspable=det["is_graspable"],
            )
        )
    return detections
