from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VlmScene:
    scene: str
    detections: list[VlmDetection]


@dataclass(frozen=True)
class VlmDetection:
    handle: str
    label: str
    # Raw bounding box as returned by the model: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]
    # Centroid derived from bbox midpoint
    centroid: tuple[float, float]
    is_graspable: bool


def parse_vlm_response(response: dict) -> VlmScene:
    """
    Parse a raw VLM JSON response into a VlmScene.

    Coordinates are stored as-is from the model (no normalisation).
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
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                centroid=(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                ),
                is_graspable=det.get("is_graspable", True),
            )
        )
    return VlmScene(scene=scene, detections=detections)
