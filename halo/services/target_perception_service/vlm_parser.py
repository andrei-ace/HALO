from __future__ import annotations

from dataclasses import dataclass

# VLM models (e.g. qwen3-vl) emit coordinates as integers in [0, 1000].
# This module normalises them to [0.0, 1.0] before any downstream use.

_VLM_COORD_MAX = 1000.0


@dataclass(frozen=True)
class VlmScene:
    scene: str
    detections: list[VlmDetection]


@dataclass(frozen=True)
class VlmDetection:
    handle: str
    label: str
    # Normalised [0, 1] bounding box: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]
    # Normalised [0, 1] centroid derived from bbox midpoint
    centroid: tuple[float, float]
    is_graspable: bool


def parse_vlm_response(response: dict) -> VlmScene:
    """
    Parse a raw VLM JSON response into a VlmScene.

    Coordinates are expected in [0, 1000] and normalised to [0.0, 1.0].
    Centroid is computed from the bounding box midpoint.
    """
    scene = response.get("scene", "")
    detections: list[VlmDetection] = []
    for det in response.get("detections", []):
        x1, y1, x2, y2 = det["bounding_box"]
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
                centroid=(
                    (x1 + x2) / 2 / _VLM_COORD_MAX,
                    (y1 + y2) / 2 / _VLM_COORD_MAX,
                ),
                is_graspable=det.get("is_graspable", True),
            )
        )
    return VlmScene(scene=scene, detections=detections)
